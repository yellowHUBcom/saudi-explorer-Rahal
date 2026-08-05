"""Deterministic Python tools for Saudi Explorer AI.

This module contains only the itinerary and budget tools. It does not call
Gemini, ChromaDB, Streamlit, or any external service. User-facing values are
Arabic, while function names and dictionary keys remain English.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


SUPPORTED_DESTINATIONS = {
    "Riyadh",
    "Jeddah",
    "Abha",
    "Eastern Province",
}

_BUDGET_ALLOCATION = {
    "accommodation": 0.40,
    "food": 0.25,
    "transport": 0.15,
    "activities": 0.15,
    "reserve": 0.05,
}

_TIME_SLOTS = ("صباحًا", "بعد الظهر", "مساءً")


class _ContextRecord(BaseModel):
    """Normalized RAG record used internally by the itinerary tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    destination: str = Field(min_length=1)
    place_name: str = Field(min_length=1)
    category: str = "عام"
    description: str = Field(min_length=1)
    recommended_duration: str = "ساعتان"
    suitable_for: str = "جميع المسافرين"
    activity_level: str = "منخفض"
    source_name: str = "قاعدة المعرفة"
    source_section: str = "عام"
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        normalized = _normalize_destination(value)
        if normalized not in SUPPORTED_DESTINATIONS:
            raise ValueError("Unsupported destination")
        return normalized


class _ItineraryInput(BaseModel):
    """Validated input for ``create_itinerary``."""

    destination: str
    days: int = Field(ge=1, le=14)
    interests: list[str] = Field(default_factory=list, max_length=10)
    context: list[dict[str, Any]] = Field(min_length=1)

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
            item = str(value).strip().casefold()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned


class _BudgetInput(BaseModel):
    """Validated input for ``calculate_budget``."""

    budget: float = Field(gt=0)
    travelers: int | None = Field(default=None, ge=1, le=50)
    days: int | None = Field(default=None, ge=1, le=30)


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


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalize_context_record(
    raw_record: dict[str, Any],
    default_destination: str,
) -> _ContextRecord:
    """Convert a Member 1 RAG dictionary into the internal tool schema."""

    if not isinstance(raw_record, dict):
        raise ValueError("Each context item must be a dictionary")

    place_name = _first_present(
        raw_record,
        ("place_name", "title", "name", "attraction", "attraction_name"),
    )
    description = _first_present(
        raw_record,
        ("description", "content", "text", "document", "chunk"),
    )

    if not place_name or not description:
        raise ValueError("Context item requires place_name and description")

    score_value = _first_present(
        raw_record,
        ("relevance_score", "score", "similarity"),
    )
    try:
        relevance_score = float(score_value) if score_value is not None else 0.0
    except (TypeError, ValueError):
        relevance_score = 0.0
    relevance_score = max(0.0, min(1.0, relevance_score))

    return _ContextRecord.model_validate(
        {
            "destination": _first_present(raw_record, ("destination", "city"))
            or default_destination,
            "place_name": str(place_name),
            "category": _first_present(raw_record, ("category", "type"))
            or "عام",
            "description": str(description),
            "recommended_duration": _first_present(
                raw_record,
                (
                    "recommended_duration",
                    "recommended_duration_hours",
                    "duration",
                ),
            )
            or "ساعتان",
            "suitable_for": _first_present(
                raw_record,
                ("suitable_for", "audience", "traveler_type"),
            )
            or "جميع المسافرين",
            "activity_level": _first_present(
                raw_record,
                ("activity_level", "effort_level"),
            )
            or "منخفض",
            "source_name": _first_present(
                raw_record,
                ("source_name", "source", "source_title"),
            )
            or "قاعدة المعرفة",
            "source_section": _first_present(
                raw_record,
                ("source_section", "section", "category"),
            )
            or "عام",
            "relevance_score": relevance_score,
        }
    )


def _duration_hours(duration_text: str) -> float:
    translated = str(duration_text).translate(
        str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    )
    match = re.search(r"(\d+(?:\.\d+)?)", translated)
    if not match:
        arabic_words = {
            "نصف ساعة": 0.5,
            "ساعة": 1.0,
            "ساعتان": 2.0,
            "ساعتين": 2.0,
            "ثلاث ساعات": 3.0,
            "أربع ساعات": 4.0,
            "اربع ساعات": 4.0,
        }
        for phrase, value in arabic_words.items():
            if phrase in translated:
                return value
        return 2.0
    value = float(match.group(1))
    return max(0.5, min(value, 12.0))


def _interest_score(record: _ContextRecord, interests: list[str]) -> float:
    searchable = " ".join(
        [
            record.place_name,
            record.category,
            record.description,
            record.suitable_for,
            record.activity_level,
        ]
    ).casefold()
    matches = sum(1 for interest in interests if interest in searchable)
    return (record.relevance_score * 10.0) + (matches * 3.0)


def _source_from_record(record: _ContextRecord) -> dict[str, str]:
    return {
        "place_name": record.place_name,
        "source_name": record.source_name,
        "source_section": record.source_section,
    }


def create_itinerary(
    destination: str,
    days: int,
    interests: list[str] | None,
    context: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a grounded Arabic day-by-day itinerary from RAG records.

    The function never invents attractions. It uses only valid context records
    that match the selected destination.

    Args:
        destination: One of the four supported destinations.
        days: Number of trip days, from 1 to 14.
        interests: Optional user interests, preferably written in Arabic.
        context: RAG records returned by ``retrieve_context``.

    Returns:
        A dictionary containing itinerary, sources, and Arabic warnings.

    Raises:
        ValueError: If inputs are invalid or no usable destination records exist.
    """

    try:
        tool_input = _ItineraryInput.model_validate(
            {
                "destination": destination,
                "days": days,
                "interests": interests or [],
                "context": context,
            }
        )
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc)) from exc

    normalized_records: list[_ContextRecord] = []
    for raw_record in tool_input.context:
        try:
            record = _normalize_context_record(
                raw_record,
                default_destination=tool_input.destination,
            )
        except (ValidationError, ValueError, TypeError):
            continue
        if record.destination == tool_input.destination:
            normalized_records.append(record)

    if not normalized_records:
        raise ValueError(
            "لا توجد أماكن مسترجعة صالحة للوجهة المختارة في قاعدة المعرفة."
        )

    unique_records: dict[str, _ContextRecord] = {}
    for record in normalized_records:
        key = record.place_name.casefold()
        previous = unique_records.get(key)
        if previous is None or record.relevance_score > previous.relevance_score:
            unique_records[key] = record

    ranked_records = sorted(
        unique_records.values(),
        key=lambda item: _interest_score(item, tool_input.interests),
        reverse=True,
    )

    itinerary: list[dict[str, Any]] = []
    total_records = len(ranked_records)
    base_count, extra_count = divmod(total_records, tool_input.days)
    record_index = 0

    for day_number in range(1, tool_input.days + 1):
        activities: list[dict[str, Any]] = []
        day_record_count = base_count + (1 if day_number <= extra_count else 0)
        day_record_count = min(day_record_count, len(_TIME_SLOTS))

        for time_slot in _TIME_SLOTS[:day_record_count]:
            record = ranked_records[record_index]
            record_index += 1
            activities.append(
                {
                    "time": time_slot,
                    "place_name": record.place_name,
                    "category": record.category,
                    "description": record.description,
                    "recommended_duration": record.recommended_duration,
                    "duration_hours": _duration_hours(
                        record.recommended_duration
                    ),
                    "suitable_for": record.suitable_for,
                    "activity_level": record.activity_level,
                    "source_name": record.source_name,
                    "source_section": record.source_section,
                }
            )

        day_note = None
        if not activities:
            day_note = (
                "لا تتوفر أماكن إضافية موثوقة في قاعدة المعرفة لهذا اليوم."
            )

        itinerary.append(
            {
                "day": day_number,
                "activities": activities,
                "note": day_note,
            }
        )

    warnings: list[str] = []
    if tool_input.days > len(ranked_records):
        warnings.append(
            "عدد أيام الرحلة أكبر من عدد الأماكن المتاحة حاليًا في قاعدة المعرفة."
        )

    sources = [_source_from_record(record) for record in ranked_records]

    return {
        "destination": tool_input.destination,
        "itinerary": itinerary,
        "sources": sources,
        "warnings": warnings,
    }


def calculate_budget(
    budget: float,
    travelers: int | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    """Allocate a user-provided total budget across trip categories.

    This tool does not use live prices. It applies transparent project-defined
    percentages to the total budget entered by the user.

    Args:
        budget: Total available budget in Saudi riyals.
        travelers: Optional number of travelers for per-person information.
        days: Optional number of days for per-day information.

    Returns:
        A structured budget allocation dictionary with Arabic user-facing text.

    Raises:
        ValueError: If budget, travelers, or days are invalid.
    """

    try:
        tool_input = _BudgetInput.model_validate(
            {
                "budget": budget,
                "travelers": travelers,
                "days": days,
            }
        )
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc)) from exc

    allocations = {
        category: round(tool_input.budget * percentage, 2)
        for category, percentage in _BUDGET_ALLOCATION.items()
    }

    result: dict[str, Any] = {
        "currency": "SAR",
        "total_budget": round(tool_input.budget, 2),
        **allocations,
        "allocation_percentages": {
            category: int(percentage * 100)
            for category, percentage in _BUDGET_ALLOCATION.items()
        },
        "disclaimer": (
            "هذا توزيع تخطيطي مبني على افتراضات المشروع، وليس أسعار سفر لحظية."
        ),
    }

    if tool_input.travelers is not None:
        result["travelers"] = tool_input.travelers
        result["budget_per_traveler"] = round(
            tool_input.budget / tool_input.travelers,
            2,
        )

    if tool_input.days is not None:
        result["days"] = tool_input.days
        result["budget_per_day"] = round(
            tool_input.budget / tool_input.days,
            2,
        )

    if tool_input.travelers is not None and tool_input.days is not None:
        result["budget_per_traveler_per_day"] = round(
            tool_input.budget / (tool_input.travelers * tool_input.days),
            2,
        )

    return result


def _format_validation_error(error: ValidationError) -> str:
    messages: list[str] = []
    for item in error.errors():
        field_name = ".".join(str(part) for part in item.get("loc", []))
        message = item.get("msg", "Invalid value")
        messages.append(f"{field_name}: {message}" if field_name else message)
    return "; ".join(messages)
