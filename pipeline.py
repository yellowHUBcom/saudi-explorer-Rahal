"""Integration pipeline for Rahhal (رحّال).

This module connects the project stages without containing UI code:

    optional Vision -> RAG retrieval -> Agent -> Python tools -> shared output

Public contract:

    run_pipeline(input_payload: dict) -> dict
"""

from __future__ import annotations

from typing import Any, Callable

from agent import run_agent


SUPPORTED_DESTINATIONS = {
    "Riyadh",
    "Jeddah",
    "Abha",
    "Eastern Province",
}

_DESTINATION_ALIASES = {
    "riyadh": "Riyadh",
    "الرياض": "Riyadh",
    "jeddah": "Jeddah",
    "جدة": "Jeddah",
    "جده": "Jeddah",
    "abha": "Abha",
    "أبها": "Abha",
    "ابها": "Abha",
    "eastern province": "Eastern Province",
    "المنطقة الشرقية": "Eastern Province",
    "الشرقية": "Eastern Province",
}

_SHARED_OUTPUT_KEYS = {
    "status",
    "destination",
    "answer",
    "itinerary",
    "budget",
    "sources",
    "warnings",
    "tools_used",
    "error",
}


def _normalize_destination(value: Any) -> str | None:
    """Return one exact internal destination ID, or ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in SUPPORTED_DESTINATIONS:
        return text
    return _DESTINATION_ALIASES.get(text.casefold())


def _output(
    *,
    status: str,
    destination: str = "",
    answer: str = "",
    itinerary: list[dict[str, Any]] | None = None,
    budget: dict[str, Any] | None = None,
    sources: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    tools_used: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the exact shared output schema for success and failure paths."""

    return {
        "status": status,
        "destination": destination,
        "answer": answer,
        "itinerary": itinerary or [],
        "budget": budget or {},
        "sources": sources or [],
        "warnings": warnings or [],
        "tools_used": tools_used or [],
        "error": error,
    }


def _error(
    answer: str,
    error_code: str,
    *,
    destination: str = "",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return _output(
        status="error",
        destination=destination,
        answer=answer,
        warnings=warnings,
        error=error_code,
    )


def _retrieve_context(
    query: str,
    destination: str,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Lazy RAG import keeps the pipeline testable without loading the model."""

    from rag import retrieve_context

    return retrieve_context(query=query, destination=destination, top_k=top_k)


def _identify_landmark(image: Any) -> dict[str, Any]:
    """Lazy Vision import; Vision remains owned by the UI/Vision member."""

    from vision import identify_landmark

    return identify_landmark(image)


def _validate_optional_positive_number(
    value: Any,
    *,
    integer: bool,
) -> int | float | None:
    """Normalize optional values while leaving detailed validation to Agent/Tools."""

    if value is None or value == "":
        return None

    if isinstance(value, bool):
        raise ValueError

    try:
        numeric = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError from exc

    if numeric <= 0:
        raise ValueError
    return numeric


def _normalize_agent_result(
    result: Any,
    *,
    destination: str,
    pipeline_warnings: list[str],
) -> dict[str, Any]:
    """Protect the UI from malformed or incomplete downstream output."""

    if not isinstance(result, dict):
        return _error(
            "أعاد وكيل الذكاء الاصطناعي نتيجة غير صالحة.",
            "INVALID_AGENT_OUTPUT",
            destination=destination,
            warnings=pipeline_warnings,
        )

    normalized = _output(
        status=str(result.get("status") or "error"),
        destination=str(result.get("destination") or destination),
        answer=str(result.get("answer") or ""),
        itinerary=(
            result.get("itinerary")
            if isinstance(result.get("itinerary"), list)
            else []
        ),
        budget=(
            result.get("budget")
            if isinstance(result.get("budget"), dict)
            else {}
        ),
        sources=(
            result.get("sources")
            if isinstance(result.get("sources"), list)
            else []
        ),
        warnings=[
            *pipeline_warnings,
            *(
                result.get("warnings")
                if isinstance(result.get("warnings"), list)
                else []
            ),
        ],
        tools_used=(
            result.get("tools_used")
            if isinstance(result.get("tools_used"), list)
            else []
        ),
        error=(str(result["error"]) if result.get("error") else None),
    )

    # Defensive assertion during development: every path has the same contract.
    assert set(normalized) == _SHARED_OUTPUT_KEYS
    return normalized


def run_pipeline(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Run the Rahhal integration pipeline and return the shared output schema.

    Expected payload keys:
        question, destination, image, days, travelers, budget, interests

    The Pipeline does not invent tourism content. RAG evidence is passed to the
    Agent, which decides whether to answer directly or execute one/both tools.
    """

    if not isinstance(input_payload, dict):
        return _error(
            "تعذّر قراءة بيانات الطلب.",
            "INVALID_INPUT_PAYLOAD",
        )

    image = input_payload.get("image")
    question = str(input_payload.get("question") or "").strip()
    destination = _normalize_destination(input_payload.get("destination"))
    warnings: list[str] = []

    # Support image-only requests without changing vision.py or app.py.
    if not question and image is not None:
        question = "ما هذا المعلم؟ وما أبرز المعلومات السياحية المتاحة عنه؟"

    if not question:
        return _error(
            "يرجى كتابة سؤال متعلق بالرحلة.",
            "EMPTY_QUESTION",
        )

    if image is not None:
        try:
            vision_result = _identify_landmark(image)
        except Exception:
            vision_result = {
                "status": "error",
                "destination": None,
                "error": "VISION_FAILURE",
            }

        if not isinstance(vision_result, dict):
            vision_result = {"status": "error", "destination": None}

        vision_status = str(vision_result.get("status") or "error").casefold()
        vision_destination = _normalize_destination(
            vision_result.get("destination")
        )

        if vision_status == "supported" and vision_destination:
            if destination and destination != vision_destination:
                warnings.append(
                    "تم اعتماد الوجهة التي تعرّف عليها تحليل الصورة بدلًا من "
                    "الاختيار اليدوي."
                )
            destination = vision_destination
        elif destination:
            warnings.append(
                "لم يُعتمد تحليل الصورة، لذلك استُخدمت الوجهة المختارة يدويًا."
            )
        else:
            return _error(
                "تعذّر تحديد وجهة مدعومة من الصورة؛ يرجى اختيار الوجهة يدويًا.",
                "MANUAL_DESTINATION_REQUIRED",
                warnings=warnings,
            )

    if destination is None:
        raw_destination = input_payload.get("destination")
        if raw_destination not in (None, ""):
            return _error(
                "الوجهة المختارة غير مدعومة حاليًا.",
                "UNSUPPORTED_DESTINATION",
            )
        return _error(
            "يرجى اختيار وجهة أو رفع صورة لمعلم مدعوم.",
            "MISSING_DESTINATION",
        )

    try:
        days = _validate_optional_positive_number(
            input_payload.get("days"),
            integer=True,
        )
    except ValueError:
        return _error(
            "يرجى إدخال عدد صحيح لأيام الرحلة أكبر من صفر.",
            "INVALID_DAYS",
            destination=destination,
            warnings=warnings,
        )

    try:
        travelers = _validate_optional_positive_number(
            input_payload.get("travelers"),
            integer=True,
        )
    except ValueError:
        return _error(
            "يرجى إدخال عدد صحيح للمسافرين أكبر من صفر.",
            "INVALID_TRAVELERS",
            destination=destination,
            warnings=warnings,
        )

    try:
        budget = _validate_optional_positive_number(
            input_payload.get("budget"),
            integer=False,
        )
    except ValueError:
        return _error(
            "يرجى إدخال ميزانية صحيحة أكبر من صفر.",
            "INVALID_BUDGET",
            destination=destination,
            warnings=warnings,
        )

    raw_interests = input_payload.get("interests") or []
    if not isinstance(raw_interests, (list, tuple, set)):
        return _error(
            "يجب إرسال الاهتمامات في صورة قائمة.",
            "INVALID_INTERESTS",
            destination=destination,
            warnings=warnings,
        )
    interests = [str(item).strip() for item in raw_interests if str(item).strip()]

    try:
        context = _retrieve_context(
            query=question,
            destination=destination,
            top_k=4,
        )
    except Exception:
        return _error(
            "تعذّر الوصول إلى قاعدة المعرفة حاليًا.",
            "RAG_FAILURE",
            destination=destination,
            warnings=warnings,
        )

    if not isinstance(context, list):
        return _error(
            "أعادت قاعدة المعرفة نتيجة غير صالحة.",
            "INVALID_RAG_OUTPUT",
            destination=destination,
            warnings=warnings,
        )

    try:
        result = run_agent(
            user_query=question,
            context=context,
            preferences={
                "destination": destination,
                "days": days,
                "travelers": travelers,
                "budget": budget,
                "interests": interests,
            },
        )
    except Exception:
        return _error(
            "تعذّر على الوكيل معالجة الطلب. يرجى المحاولة مرة أخرى.",
            "AGENT_FAILURE",
            destination=destination,
            warnings=warnings,
        )

    return _normalize_agent_result(
        result,
        destination=destination,
        pipeline_warnings=warnings,
    )
