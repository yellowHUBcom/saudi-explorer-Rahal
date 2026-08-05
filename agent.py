"""AI agent and tool selection for Saudi Explorer AI.

Public contract required by PROJECT_SPEC.md:

    run_agent(user_query, context, preferences) -> dict

The module is compatible with Member 1's exact ``retrieve_context`` output,
selects the minimum required tools, and returns the fixed shared output schema.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tools import (
    SUPPORTED_DESTINATIONS,
    calculate_budget,
    create_itinerary,
    normalize_rag_record,
)


class _ToolName(str, Enum):
    CREATE_ITINERARY = "create_itinerary"
    CALCULATE_BUDGET = "calculate_budget"


class _AgentAction(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    ITINERARY = "itinerary"
    BUDGET = "budget"
    ITINERARY_AND_BUDGET = "itinerary_and_budget"


class _AgentDecision(BaseModel):
    """Structured Gemini output used for tool routing."""

    action: _AgentAction
    tools_to_use: list[_ToolName] = Field(default_factory=list, max_length=2)
    reason: str = Field(min_length=3, max_length=500)

    @field_validator("tools_to_use")
    @classmethod
    def reject_duplicate_tools(cls, values: list[_ToolName]) -> list[_ToolName]:
        if len(values) != len(set(values)):
            raise ValueError("Duplicate tools are not allowed")
        return values


class _Preferences(BaseModel):
    """Validated preferences received from pipeline.py."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    destination: str = Field(min_length=1)
    days: int | None = Field(default=None, ge=1, le=14)
    travelers: int | None = Field(default=None, ge=1, le=50)
    budget: float | None = Field(default=None, gt=0)
    interests: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        normalized = _normalize_destination(value)
        if normalized not in SUPPORTED_DESTINATIONS:
            raise ValueError("Unsupported destination")
        return normalized

    @field_validator("interests")
    @classmethod
    def clean_interests(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = str(value).strip()
            if item and item.casefold() not in {
                existing.casefold() for existing in cleaned
            }:
                cleaned.append(item)
        return cleaned


class _SharedOutput(BaseModel):
    """Exact shared output schema from PROJECT_SPEC.md."""

    model_config = ConfigDict(extra="forbid")

    status: str = "success"
    destination: str = ""
    answer: str = ""
    itinerary: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    error: str | None = None


_AGENT_SYSTEM_PROMPT = """أنت وكيل التوجيه في تطبيق رحّال.
مهمتك اختيار أقل عدد من الأدوات اللازمة لتنفيذ طلب المستخدم.

الأدوات المتاحة فقط:
- create_itinerary: لإنشاء خطة رحلة من نتائج RAG الموثوقة.
- calculate_budget: لتوزيع ميزانية إجمالية أدخلها المستخدم.

القواعد:
- استخدم الأداتين إذا طلب المستخدم خطة وميزانية معًا، أو طلب خطة وكانت
  الميزانية موجودة في preferences.
- لا تستخدم أي أداة لسؤال سياحي عادي يمكن الإجابة عنه من سياق RAG.
- لا تستخدم الأدوات للحجوزات أو الفنادق أو الطيران أو الطقس المباشر أو الأسعار
  اللحظية؛ هذه الطلبات خارج نطاق المشروع.
- أعد النتيجة وفق مخطط القرار المنظم فقط.
- يجب أن تبقى قيم action وtools_to_use بالإنجليزية كما هي في المخطط.
"""

_FINAL_ANSWER_SYSTEM_PROMPT = """أنت مساعد سفر سعودي باسم رحّال.
أجب باللغة العربية الفصحى الواضحة والمناسبة لواجهة موقع سياحي.
استخدم فقط سياق RAG ونتائج أدوات Python المرسلة إليك.
لا تخترع أماكن أو أسعارًا أو أوقات عمل أو حقائق غير موجودة في الأدلة.
أبق أسماء الأماكن كما وردت في السياق، وقدم الإجابة بطريقة موجزة ومنظمة.
إذا كانت هناك نتيجة ميزانية، وضح أنها توزيع للمبلغ الذي أدخله المستخدم وليست
أسعارًا لحظية. وإذا كانت الأدلة غير كافية، صرّح بذلك بوضوح.
"""

_ITINERARY_TERMS = (
    "itinerary",
    "schedule",
    "plan a trip",
    "plan my trip",
    "trip plan",
    "day-by-day",
    "خطة",
    "خطة رحلة",
    "خطة سفر",
    "جدول",
    "جدول رحلة",
    "برنامج رحلة",
    "برنامج سياحي",
    "رتب لي",
    "رتّب لي",
    "خطط",
    "خطّط",
    "سوي لي رحلة",
    "سو لي رحلة",
    "أنشئ لي رحلة",
    "اقترح لي برنامج",
)

_BUDGET_TERMS = (
    "budget",
    "allocate budget",
    "budget allocation",
    "ميزانية",
    "وزع ميزانيتي",
    "وزّع ميزانيتي",
    "وزع الميزانية",
    "وزّع الميزانية",
    "توزيع الميزانية",
    "قسم الميزانية",
    "قسّم الميزانية",
    "تخصيص الميزانية",
    "احسب ميزانية الرحلة",
    "خطط للميزانية",
)

_TRIP_TERMS = (
    "trip",
    "travel",
    "visit",
    "رحلة",
    "سفر",
    "زيارة",
    "سياحة",
)

_OUT_OF_SCOPE_TERMS = (
    "فندق",
    "فنادق",
    "حجز فندق",
    "حجوزات",
    "احجز",
    "حجز طيران",
    "تذاكر طيران",
    "رحلات طيران",
    "الطقس",
    "درجة الحرارة",
    "الجو اليوم",
    "سعر التذكرة",
    "أسعار التذاكر",
    "سعر الدخول",
    "تكلفة الدخول",
    "كم سعر",
    "كم تكلفة",
    "hotel",
    "booking",
    "book a hotel",
    "flight",
    "live weather",
    "temperature today",
    "ticket price",
    "live price",
)

_ARABIC_DAY_WORDS = {
    "يوم واحد": 1,
    "يوماً واحداً": 1,
    "يومًا واحدًا": 1,
    "يومين": 2,
    "يومان": 2,
    "ثلاثة أيام": 3,
    "ثلاث ايام": 3,
    "ثلاث أيام": 3,
    "أربعة أيام": 4,
    "اربعة ايام": 4,
    "أربع أيام": 4,
    "خمسة أيام": 5,
    "خمس ايام": 5,
    "ستة أيام": 6,
    "ست ايام": 6,
    "سبعة أيام": 7,
    "سبع ايام": 7,
    "ثمانية أيام": 8,
    "ثمان ايام": 8,
    "تسعة أيام": 9,
    "تسع ايام": 9,
    "عشرة أيام": 10,
    "عشر ايام": 10,
    "أحد عشر يومًا": 11,
    "احد عشر يوما": 11,
    "اثنا عشر يومًا": 12,
    "اثنا عشر يوما": 12,
    "ثلاثة عشر يومًا": 13,
    "ثلاثة عشر يوما": 13,
    "أربعة عشر يومًا": 14,
    "اربعة عشر يوما": 14,
}

_ARABIC_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize_destination(value: str) -> str:
    aliases = {
        "riyadh": "Riyadh",
        "الرياض": "Riyadh",
        "jeddah": "Jeddah",
        "جدة": "Jeddah",
        "abha": "Abha",
        "أبها": "Abha",
        "ابها": "Abha",
        "eastern province": "Eastern Province",
        "eastern": "Eastern Province",
        "المنطقة الشرقية": "Eastern Province",
        "الشرقية": "Eastern Province",
    }
    text = str(value).strip()
    return aliases.get(text.casefold(), text)


def _error_output(
    message: str,
    *,
    destination: str = "",
    error_code: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    output = _SharedOutput(
        status="error",
        destination=destination,
        answer=message,
        warnings=warnings or [],
        error=error_code,
    )
    return output.model_dump(mode="json")


def _format_validation_error(error: ValidationError) -> str:
    messages: list[str] = []
    for item in error.errors():
        field_name = ".".join(str(part) for part in item.get("loc", []))
        message = item.get("msg", "Invalid value")
        messages.append(f"{field_name}: {message}" if field_name else message)
    return "; ".join(messages)


def _extract_days_from_query(user_query: str) -> int | None:
    normalized_query = user_query.translate(_ARABIC_DIGIT_TRANSLATION).casefold()

    for phrase, value in sorted(
        _ARABIC_DAY_WORDS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if phrase.casefold() in normalized_query:
            return value

    patterns = (
        r"\b(\d{1,2})\s*(?:day|days)\b",
        r"(\d{1,2})\s*(?:يوم|أيام|ايام)",
        r"\b(one|two|three|four|five|six|seven)\s*[- ]?day\b",
    )
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
    }
    for pattern in patterns:
        match = re.search(pattern, normalized_query, flags=re.IGNORECASE)
        if not match:
            continue
        raw_value = match.group(1).casefold()
        value = word_numbers.get(
            raw_value,
            int(raw_value) if raw_value.isdigit() else 0,
        )
        if 1 <= value <= 14:
            return value
    return None


def _is_budget_allocation_request(user_query: str) -> bool:
    text = user_query.casefold()
    return any(term in text for term in _BUDGET_TERMS)


def _is_out_of_scope(user_query: str) -> bool:
    if _is_budget_allocation_request(user_query):
        return False
    text = user_query.casefold()
    return any(term in text for term in _OUT_OF_SCOPE_TERMS)


def _local_decision(user_query: str, preferences: _Preferences) -> _AgentDecision:
    text = user_query.casefold()
    wants_itinerary = any(term in text for term in _ITINERARY_TERMS)
    wants_budget = any(term in text for term in _BUDGET_TERMS)
    mentions_trip = any(term in text for term in _TRIP_TERMS)

    if wants_itinerary and (wants_budget or preferences.budget is not None):
        return _AgentDecision(
            action=_AgentAction.ITINERARY_AND_BUDGET,
            tools_to_use=[
                _ToolName.CREATE_ITINERARY,
                _ToolName.CALCULATE_BUDGET,
            ],
            reason="الطلب يحتاج إلى خطة رحلة وتوزيع ميزانية.",
        )

    if wants_itinerary:
        return _AgentDecision(
            action=_AgentAction.ITINERARY,
            tools_to_use=[_ToolName.CREATE_ITINERARY],
            reason="الطلب يتضمن إنشاء خطة رحلة مخصصة.",
        )

    if wants_budget:
        return _AgentDecision(
            action=_AgentAction.BUDGET,
            tools_to_use=[_ToolName.CALCULATE_BUDGET],
            reason="الطلب يتضمن توزيع الميزانية الإجمالية.",
        )

    if mentions_trip and preferences.days is not None and preferences.budget is not None:
        return _AgentDecision(
            action=_AgentAction.ITINERARY_AND_BUDGET,
            tools_to_use=[
                _ToolName.CREATE_ITINERARY,
                _ToolName.CALCULATE_BUDGET,
            ],
            reason="تم توفير عدد الأيام والميزانية لطلب رحلة كامل.",
        )

    return _AgentDecision(
        action=_AgentAction.DIRECT_ANSWER,
        tools_to_use=[],
        reason="يمكن الإجابة عن الطلب مباشرة من سياق RAG.",
    )


def _get_gemini_client() -> Any:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Add it to requirements.txt."
        ) from exc
    return genai.Client(api_key=api_key)


def _gemini_decision(
    user_query: str,
    preferences: _Preferences,
) -> _AgentDecision:
    from google.genai import types

    client = _get_gemini_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "user_query": user_query,
        "preferences": preferences.model_dump(mode="json"),
        "available_tools": [
            {
                "name": "create_itinerary",
                "purpose": "إنشاء خطة رحلة من سجلات RAG المسترجعة فقط.",
            },
            {
                "name": "calculate_budget",
                "purpose": "توزيع ميزانية إجمالية موجبة أدخلها المستخدم.",
            },
        ],
    }
    response = client.models.generate_content(
        model=model_name,
        contents=json.dumps(payload, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=_AGENT_SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=_AgentDecision,
        ),
    )

    if isinstance(response.parsed, _AgentDecision):
        return response.parsed
    if isinstance(response.parsed, dict):
        return _AgentDecision.model_validate(response.parsed)
    if response.text:
        return _AgentDecision.model_validate_json(response.text)
    raise RuntimeError("Gemini returned an empty tool decision")


def decide_tools(
    user_query: str,
    preferences: dict[str, Any],
) -> dict[str, Any]:
    """Return a structured decision describing which tools should run."""

    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must not be empty")

    try:
        validated_preferences = _Preferences.model_validate(preferences)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc)) from exc

    if _is_out_of_scope(user_query):
        return {
            "action": "direct_answer",
            "tools_to_use": [],
            "reason": "الطلب خارج نطاق النسخة الحالية من المشروع.",
        }

    if os.getenv("GEMINI_API_KEY"):
        try:
            decision = _gemini_decision(user_query, validated_preferences)
            return decision.model_dump(mode="json")
        except Exception:
            pass

    return _local_decision(
        user_query,
        validated_preferences,
    ).model_dump(mode="json")


def _normalize_context(
    context: list[dict[str, Any]],
    destination: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_record in context:
        try:
            record = normalize_rag_record(raw_record, destination)
        except (ValueError, ValidationError, TypeError):
            continue
        if record["destination"] == destination:
            normalized.append(record)
    return normalized


def _collect_sources(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_sources: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in context:
        source = {
            "place_name": record.get("place_name", ""),
            "source_name": record.get("source_name", "قاعدة المعرفة"),
            "source_section": record.get("source_section", "عام"),
            "source_url": record.get("source_url", ""),
        }
        key = (
            source["place_name"],
            source["source_name"],
            source["source_section"],
            source["source_url"],
        )
        unique_sources[key] = source
    return list(unique_sources.values())


def _fallback_direct_answer(context: list[dict[str, Any]]) -> str:
    summaries = [
        f"• {record['place_name']}: {record['description']}"
        for record in context[:4]
    ]
    return "بحسب المعلومات المتاحة في قاعدة المعرفة:\n\n" + "\n\n".join(
        summaries
    )


def _fallback_tool_answer(tools_used: list[str]) -> str:
    if tools_used == ["create_itinerary"]:
        return "تم إنشاء خطة الرحلة اعتمادًا على المعالم المسترجعة من قاعدة المعرفة."
    if tools_used == ["calculate_budget"]:
        return "تم توزيع الميزانية الإجمالية على فئات الرحلة بنجاح."
    if tools_used == ["create_itinerary", "calculate_budget"]:
        return "تم إنشاء خطة الرحلة وتوزيع الميزانية بنجاح."
    return "تم تنفيذ الطلب بنجاح."


def _generate_final_answer(
    user_query: str,
    context: list[dict[str, Any]],
    itinerary: list[dict[str, Any]],
    budget: dict[str, Any],
    tools_used: list[str],
) -> tuple[str, str | None]:
    """Generate an Arabic grounded answer and return an optional warning."""

    if not os.getenv("GEMINI_API_KEY"):
        if tools_used:
            return (
                _fallback_tool_answer(tools_used),
                "مفتاح Gemini غير متوفر؛ تم استخدام الاستجابة المحلية البديلة.",
            )
        return (
            _fallback_direct_answer(context),
            "مفتاح Gemini غير متوفر؛ تم عرض المعلومات المسترجعة مباشرة.",
        )

    try:
        from google.genai import types

        client = _get_gemini_client()
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        payload = {
            "user_query": user_query,
            "retrieved_context": context,
            "itinerary": itinerary,
            "budget": budget,
            "tools_used": tools_used,
        }
        response = client.models.generate_content(
            model=model_name,
            contents=json.dumps(payload, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=_FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty final response")
        return response.text.strip(), None
    except Exception:
        fallback = (
            _fallback_tool_answer(tools_used)
            if tools_used
            else _fallback_direct_answer(context)
        )
        return (
            fallback,
            "تعذّر الاتصال بخدمة Gemini؛ تم استخدام استجابة محلية آمنة.",
        )


def run_agent(
    user_query: str,
    context: list[dict[str, Any]],
    preferences: dict[str, Any],
) -> dict[str, Any]:
    """Select and execute tools, then return the shared output schema."""

    if not isinstance(user_query, str) or not user_query.strip():
        return _error_output(
            "يرجى كتابة سؤال متعلق بالرحلة.",
            error_code="INVALID_QUESTION",
        )

    try:
        validated_preferences = _Preferences.model_validate(preferences)
    except ValidationError as exc:
        message = _format_validation_error(exc)
        error_code = "INVALID_PREFERENCES"
        if "budget" in message:
            message = "يرجى إدخال ميزانية صحيحة أكبر من صفر."
            error_code = "INVALID_BUDGET"
        elif "days" in message:
            message = "يرجى إدخال عدد صحيح لأيام الرحلة."
            error_code = "INVALID_DAYS"
        elif "travelers" in message:
            message = "يرجى إدخال عدد صحيح للمسافرين."
            error_code = "INVALID_TRAVELERS"
        elif "destination" in message:
            message = "الوجهة المختارة غير مدعومة حاليًا."
            error_code = "UNSUPPORTED_DESTINATION"
        return _error_output(message, error_code=error_code)

    destination = validated_preferences.destination

    if _is_out_of_scope(user_query):
        return _error_output(
            "هذا الطلب خارج نطاق النسخة الحالية من رحّال، والتي تركز على "
            "معلومات المعالم، وتخطيط الرحلات، وتوزيع الميزانية.",
            destination=destination,
            error_code="OUT_OF_SCOPE",
        )

    if not isinstance(context, list):
        return _error_output(
            "يجب أن تكون نتائج الاسترجاع قائمة من السجلات.",
            destination=destination,
            error_code="INVALID_CONTEXT",
        )

    api_warning: str | None = None
    try:
        if os.getenv("GEMINI_API_KEY"):
            try:
                decision = _gemini_decision(user_query, validated_preferences)
            except Exception:
                decision = _local_decision(user_query, validated_preferences)
                api_warning = (
                    "تعذّر استخدام Gemini لاختيار الأدوات؛ تم استخدام التوجيه "
                    "المحلي البديل."
                )
        else:
            decision = _local_decision(user_query, validated_preferences)
    except Exception:
        return _error_output(
            "تعذّر على وكيل الذكاء الاصطناعي معالجة الطلب. يرجى المحاولة مرة أخرى.",
            destination=destination,
            error_code="AGENT_FAILURE",
        )

    normalized_context = _normalize_context(context, destination)
    requires_context = (
        decision.action in {_AgentAction.DIRECT_ANSWER, _AgentAction.ITINERARY,
                            _AgentAction.ITINERARY_AND_BUDGET}
    )
    if requires_context and not normalized_context:
        return _error_output(
            "لا تتوفر معلومات كافية في قاعدة المعرفة للإجابة عن هذا الطلب.",
            destination=destination,
            error_code="EMPTY_RETRIEVAL",
        )

    days = validated_preferences.days or _extract_days_from_query(user_query)
    itinerary: list[dict[str, Any]] = []
    budget_result: dict[str, Any] = {}
    sources = _collect_sources(normalized_context)
    warnings: list[str] = []
    tools_used: list[str] = []

    if api_warning:
        warnings.append(api_warning)

    if _ToolName.CREATE_ITINERARY in decision.tools_to_use:
        if days is None or days <= 0:
            return _error_output(
                "يرجى إدخال عدد صحيح لأيام الرحلة قبل إنشاء الخطة.",
                destination=destination,
                error_code="INVALID_DAYS",
                warnings=warnings,
            )
        try:
            itinerary_result = create_itinerary(
                destination=destination,
                days=days,
                interests=validated_preferences.interests,
                context=normalized_context,
            )
            itinerary = itinerary_result["itinerary"]
            sources = itinerary_result["sources"]
            warnings.extend(itinerary_result.get("warnings", []))
            tools_used.append("create_itinerary")
        except (ValueError, TypeError):
            return _error_output(
                "تعذّر إنشاء خطة الرحلة باستخدام المعلومات المتاحة.",
                destination=destination,
                error_code="ITINERARY_TOOL_ERROR",
                warnings=warnings,
            )

    if _ToolName.CALCULATE_BUDGET in decision.tools_to_use:
        if validated_preferences.budget is None:
            return _error_output(
                "يرجى إدخال ميزانية إجمالية صحيحة أكبر من صفر.",
                destination=destination,
                error_code="INVALID_BUDGET",
                warnings=warnings,
            )
        try:
            budget_result = calculate_budget(
                budget=validated_preferences.budget,
                travelers=validated_preferences.travelers,
                days=days,
            )
            tools_used.append("calculate_budget")
        except (ValueError, TypeError):
            return _error_output(
                "تعذّر حساب توزيع الميزانية.",
                destination=destination,
                error_code="BUDGET_TOOL_ERROR",
                warnings=warnings,
            )

    answer, final_answer_warning = _generate_final_answer(
        user_query=user_query,
        context=normalized_context,
        itinerary=itinerary,
        budget=budget_result,
        tools_used=tools_used,
    )
    if final_answer_warning and final_answer_warning not in warnings:
        warnings.append(final_answer_warning)

    output = _SharedOutput(
        status="success",
        destination=destination,
        answer=answer,
        itinerary=itinerary,
        budget=budget_result,
        sources=sources,
        warnings=warnings,
        tools_used=tools_used,
        error=None,
    )
    return output.model_dump(mode="json")
