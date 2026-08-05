"""AI agent and tool selection for Saudi Explorer AI.

Public contract required by PROJECT_SPEC.md:

    run_agent(user_query, context, preferences) -> dict

The module decides which tools are required, executes the selected tools, and
returns the exact shared output schema. It contains no Streamlit, Vision, RAG,
or ChromaDB implementation.
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
    """Structured Gemini output for agent routing."""

    action: _AgentAction
    tools_to_use: list[_ToolName] = Field(default_factory=list, max_length=2)
    reason: str = Field(min_length=3, max_length=500)

    @field_validator("tools_to_use")
    @classmethod
    def reject_duplicate_tools(
        cls,
        values: list[_ToolName],
    ) -> list[_ToolName]:
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


_AGENT_SYSTEM_PROMPT = """You are the routing agent for Saudi Explorer AI.
Choose the minimum tools needed for the user's request.

Available tools:
- create_itinerary: use for a trip plan, itinerary, schedule, or day-by-day plan.
- calculate_budget: use when the user asks to allocate or calculate a total budget.

Rules:
- Use both tools when the user asks for both a plan and a budget, or asks for a
  trip plan while a total budget is supplied in preferences.
- Use no tool for a normal travel question that can be answered from RAG.
- Never select any tool outside the registered tools.
- Return only the structured decision schema.
"""

_FINAL_ANSWER_SYSTEM_PROMPT = """You are Saudi Explorer AI.
Produce a concise travel response using only the supplied RAG context and Python
tool results. Never invent places, prices, opening hours, or unsupported facts.
Mention that budget values are an allocation of the user's total budget and are
not real-time prices. If evidence is insufficient, state that clearly.
"""

_ITINERARY_TERMS = (
    "itinerary",
    "schedule",
    "plan a trip",
    "plan my trip",
    "trip plan",
    "day-by-day",
    "two-day",
    "three-day",
    "خطة",
    "جدول",
    "برنامج رحلة",
    "رتب لي",
    "خطط",
)

_BUDGET_TERMS = (
    "budget",
    "allocate",
    "allocation",
    "cost",
    "how much",
    "ميزانية",
    "وزع الميزانية",
    "توزيع الميزانية",
    "تكلفة",
)

_TRIP_TERMS = ("trip", "travel", "visit", "رحلة", "سفر", "زيارة")


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


def _empty_output(destination: str = "") -> _SharedOutput:
    return _SharedOutput(destination=destination)


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
    patterns = (
        r"\b(\d{1,2})\s*(?:day|days)\b",
        r"\b(\d{1,2})\s*(?:يوم|أيام|ايام)\b",
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
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if not match:
            continue
        raw_value = match.group(1).casefold()
        value = word_numbers.get(raw_value, int(raw_value) if raw_value.isdigit() else 0)
        if 1 <= value <= 14:
            return value
    return None


def _local_decision(
    user_query: str,
    preferences: _Preferences,
) -> _AgentDecision:
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
            reason="The request needs a trip plan and a budget allocation.",
        )

    if wants_itinerary:
        return _AgentDecision(
            action=_AgentAction.ITINERARY,
            tools_to_use=[_ToolName.CREATE_ITINERARY],
            reason="The request asks for a personalized itinerary.",
        )

    if wants_budget:
        return _AgentDecision(
            action=_AgentAction.BUDGET,
            tools_to_use=[_ToolName.CALCULATE_BUDGET],
            reason="The request asks for a budget allocation.",
        )

    if mentions_trip and preferences.days is not None and preferences.budget is not None:
        return _AgentDecision(
            action=_AgentAction.ITINERARY_AND_BUDGET,
            tools_to_use=[
                _ToolName.CREATE_ITINERARY,
                _ToolName.CALCULATE_BUDGET,
            ],
            reason="Trip days and a total budget were supplied.",
        )

    return _AgentDecision(
        action=_AgentAction.DIRECT_ANSWER,
        tools_to_use=[],
        reason="The request can be answered directly from retrieved context.",
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
                "purpose": "Create a grounded itinerary from retrieved records.",
            },
            {
                "name": "calculate_budget",
                "purpose": "Allocate a positive user-provided total budget.",
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
    """Return a structured decision describing which tools should run.

    Gemini is used when ``GEMINI_API_KEY`` is available. If the API is missing
    or temporarily unavailable, a deterministic local decision is returned and
    the caller can show a friendly warning instead of crashing.
    """

    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must not be empty")

    try:
        validated_preferences = _Preferences.model_validate(preferences)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc)) from exc

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
        if not isinstance(raw_record, dict):
            continue

        place_name = _first_present(
            raw_record,
            ("place_name", "title", "name", "attraction", "attraction_name"),
        )
        description = _first_present(
            raw_record,
            ("description", "content", "text", "document", "chunk"),
        )
        record_destination = _normalize_destination(
            _first_present(raw_record, ("destination", "city")) or destination
        )

        if not place_name or not description or record_destination != destination:
            continue

        normalized.append(
            {
                **raw_record,
                "destination": record_destination,
                "place_name": str(place_name),
                "description": str(description),
                "category": _first_present(raw_record, ("category", "type"))
                or "General",
                "source_name": _first_present(
                    raw_record,
                    ("source_name", "source", "source_title"),
                )
                or "Knowledge Base",
                "source_section": _first_present(
                    raw_record,
                    ("source_section", "section", "category"),
                )
                or "General",
            }
        )
    return normalized


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _collect_sources(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_sources: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in context:
        source = {
            "place_name": record.get("place_name", ""),
            "source_name": record.get("source_name", "Knowledge Base"),
            "source_section": record.get("source_section", "General"),
        }
        key = (
            source["place_name"],
            source["source_name"],
            source["source_section"],
        )
        unique_sources[key] = source
    return list(unique_sources.values())


def _fallback_direct_answer(context: list[dict[str, Any]]) -> str:
    summaries = [
        f"{record['place_name']}: {record['description']}"
        for record in context[:4]
    ]
    return "\n\n".join(summaries)


def _fallback_tool_answer(tools_used: list[str]) -> str:
    if tools_used == ["create_itinerary"]:
        return "A grounded itinerary was created from the retrieved attractions."
    if tools_used == ["calculate_budget"]:
        return "The total budget was allocated across the trip categories."
    if tools_used == ["create_itinerary", "calculate_budget"]:
        return (
            "A grounded itinerary and a budget allocation were created "
            "successfully."
        )
    return "The request was completed successfully."


def _generate_final_answer(
    user_query: str,
    context: list[dict[str, Any]],
    itinerary: list[dict[str, Any]],
    budget: dict[str, Any],
    tools_used: list[str],
) -> tuple[str, str | None]:
    """Generate a grounded user-facing answer and return an optional warning."""

    if not os.getenv("GEMINI_API_KEY"):
        if tools_used:
            return (
                _fallback_tool_answer(tools_used),
                "Gemini API key is unavailable; a local fallback response was used.",
            )
        return (
            _fallback_direct_answer(context),
            "Gemini API key is unavailable; retrieved context was shown directly.",
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
            "Gemini API was unavailable; a safe local fallback response was used.",
        )


def run_agent(
    user_query: str,
    context: list[dict[str, Any]],
    preferences: dict[str, Any],
) -> dict[str, Any]:
    """Select and execute tools, then return the shared output schema.

    Args:
        user_query: The user's travel question.
        context: Retrieved RAG records from ``retrieve_context``.
        preferences: Destination, days, travelers, budget, and interests.

    Returns:
        A dictionary with exactly these keys: status, destination, answer,
        itinerary, budget, sources, warnings, tools_used, and error.
    """

    if not isinstance(user_query, str) or not user_query.strip():
        return _error_output(
            "Please enter a travel question.",
            error_code="INVALID_QUESTION",
        )

    try:
        validated_preferences = _Preferences.model_validate(preferences)
    except ValidationError as exc:
        message = _format_validation_error(exc)
        error_code = "INVALID_PREFERENCES"
        if "budget" in message:
            message = "Please enter a valid budget greater than zero."
            error_code = "INVALID_BUDGET"
        elif "days" in message:
            message = "Please enter a valid number of trip days."
            error_code = "INVALID_DAYS"
        elif "destination" in message:
            message = "The selected destination is not supported."
            error_code = "UNSUPPORTED_DESTINATION"
        return _error_output(message, error_code=error_code)

    destination = validated_preferences.destination

    if not isinstance(context, list):
        return _error_output(
            "Retrieved context must be a list of records.",
            destination=destination,
            error_code="INVALID_CONTEXT",
        )

    normalized_context = _normalize_context(context, destination)
    if not normalized_context:
        return _error_output(
            "There is insufficient information in the knowledge base for this "
            "request.",
            destination=destination,
            error_code="EMPTY_RETRIEVAL",
        )

    api_warning: str | None = None
    try:
        if os.getenv("GEMINI_API_KEY"):
            try:
                decision = _gemini_decision(
                    user_query,
                    validated_preferences,
                )
            except Exception:
                decision = _local_decision(
                    user_query,
                    validated_preferences,
                )
                api_warning = (
                    "Gemini tool selection was unavailable; local tool routing "
                    "was used."
                )
        else:
            decision = _local_decision(user_query, validated_preferences)
            api_warning = (
                "Gemini API key is unavailable; local tool routing was used."
            )
    except Exception as exc:
        return _error_output(
            "The AI agent could not process the request. Please try again.",
            destination=destination,
            error_code="AGENT_FAILURE",
            warnings=[str(exc)],
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
                "Please enter a valid number of trip days before generating an "
                "itinerary.",
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
        except (ValueError, TypeError) as exc:
            return _error_output(
                f"The itinerary tool could not complete the request: {exc}",
                destination=destination,
                error_code="ITINERARY_TOOL_ERROR",
                warnings=warnings,
            )

    if _ToolName.CALCULATE_BUDGET in decision.tools_to_use:
        if validated_preferences.budget is None:
            return _error_output(
                "Please enter a valid total budget greater than zero.",
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
        except (ValueError, TypeError) as exc:
            return _error_output(
                f"The budget tool could not complete the request: {exc}",
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
