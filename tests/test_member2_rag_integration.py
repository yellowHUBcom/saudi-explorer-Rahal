"""Tests for Member 2 using Member 1's exact RAG result shape."""

from __future__ import annotations

from agent import decide_tools, run_agent
from tools import calculate_budget, create_itinerary, normalize_rag_record


def _rag_record(
    place_name: str,
    category: str,
    description: str,
    duration: str,
    suitable_for: str,
    activity_level: str,
    source_name: str,
    source_section: str,
    source_url: str,
    distance: float,
) -> dict:
    text = "\n".join(
        [
            "الوجهة: Eastern Province",
            f"اسم المكان: {place_name}",
            f"التصنيف: {category}",
            f"الوصف: {description}",
            f"المدة المقترحة: {duration}",
            f"مناسب لـ: {suitable_for}",
            f"مستوى النشاط: {activity_level}",
        ]
    )
    return {
        "text": text,
        "destination": "Eastern Province",
        "place_name": place_name,
        "category": category,
        "source_name": source_name,
        "source_section": source_section,
        "source_url": source_url,
        "distance": distance,
    }


EASTERN_RAG_CONTEXT = [
    _rag_record(
        "نبذة عن المنطقة الشرقية",
        "معلومات عامة",
        "تجمع المنطقة الشرقية بين الواجهات البحرية والمراكز الثقافية الحديثة.",
        "عدة أيام",
        "جميع المسافرين",
        "متوسط",
        "روح السعودية",
        "المنطقة الشرقية",
        "https://www.visitsaudi.com/ar/eastern-province",
        0.20,
    ),
    _rag_record(
        "مركز الملك عبدالعزيز الثقافي العالمي — إثراء",
        "ثقافي",
        "مركز ثقافي في الظهران يضم معارض ومساحات تعليمية متنوعة.",
        "نصف يوم",
        "جميع المسافرين",
        "منخفض",
        "إثراء",
        "عن إثراء",
        "https://www.ithra.com/ar/about-ithra",
        0.10,
    ),
    _rag_record(
        "الواجهة البحرية بالخبر",
        "واجهة بحرية",
        "مساحة مفتوحة مناسبة للمشي والجلوس والاستمتاع بالمشهد الساحلي.",
        "3 ساعات",
        "جميع المسافرين",
        "متوسط",
        "روح السعودية",
        "الواجهة البحرية بالخبر",
        "https://www.visitsaudi.com/ar/eastern-province/attractions/khobar-seafront",
        0.18,
    ),
    _rag_record(
        "جزيرة المرجان",
        "طبيعي",
        "جزيرة ترفيهية مناسبة للتنزه والتصوير والاسترخاء بجانب البحر.",
        "ساعتان",
        "جميع المسافرين",
        "منخفض",
        "روح السعودية",
        "جزيرة المرجان",
        "https://www.visitsaudi.com/ar/eastern-province/attractions/murjan-island-in-dammam",
        0.24,
    ),
]


def _disable_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_normalizes_exact_rag_result_and_parses_document_fields():
    result = normalize_rag_record(
        EASTERN_RAG_CONTEXT[1],
        default_destination="Eastern Province",
    )
    assert result["description"].startswith("مركز ثقافي")
    assert result["recommended_duration"] == "نصف يوم"
    assert result["suitable_for"] == "جميع المسافرين"
    assert result["activity_level"] == "منخفض"
    assert result["source_url"].startswith("https://")
    assert result["relevance_score"] == 0.9


def test_overview_is_excluded_from_itinerary():
    result = create_itinerary(
        destination="Eastern Province",
        days=2,
        interests=["ثقافة", "بحر"],
        context=EASTERN_RAG_CONTEXT,
    )
    places = {
        activity["place_name"]
        for day in result["itinerary"]
        for activity in day["activities"]
    }
    assert "نبذة عن المنطقة الشرقية" not in places
    assert len(places) == 3


def test_itinerary_keeps_source_urls():
    result = create_itinerary(
        destination="Eastern Province",
        days=1,
        interests=[],
        context=EASTERN_RAG_CONTEXT,
    )
    assert all(source["source_url"] for source in result["sources"])
    assert all(
        activity["source_url"]
        for activity in result["itinerary"][0]["activities"]
    )


def test_direct_question_uses_no_tools(monkeypatch):
    _disable_gemini(monkeypatch)
    decision = decide_tools(
        "ما الأماكن الثقافية في المنطقة الشرقية؟",
        {"destination": "Eastern Province"},
    )
    assert decision["action"] == "direct_answer"
    assert decision["tools_to_use"] == []


def test_itinerary_decision(monkeypatch):
    _disable_gemini(monkeypatch)
    decision = decide_tools(
        "خطط لي رحلة عائلية لمدة يومين",
        {"destination": "Eastern Province", "days": 2},
    )
    assert decision["tools_to_use"] == ["create_itinerary"]


def test_budget_decision(monkeypatch):
    _disable_gemini(monkeypatch)
    decision = decide_tools(
        "وزع ميزانيتي على الرحلة",
        {"destination": "Eastern Province", "budget": 4000},
    )
    assert decision["tools_to_use"] == ["calculate_budget"]


def test_budget_only_can_run_when_rag_returns_empty(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "وزع ميزانيتي على الرحلة",
        [],
        {
            "destination": "Eastern Province",
            "budget": 4000,
            "travelers": 2,
        },
    )
    assert result["status"] == "success"
    assert result["tools_used"] == ["calculate_budget"]
    assert result["sources"] == []


def test_full_request_uses_both_tools_with_exact_rag_shape(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "خطط لي رحلة عائلية لمدة يومين ووزع الميزانية",
        EASTERN_RAG_CONTEXT,
        {
            "destination": "Eastern Province",
            "days": 2,
            "travelers": 3,
            "budget": 4500,
            "interests": ["عائلات", "ثقافة", "بحر"],
        },
    )
    assert result["status"] == "success"
    assert result["tools_used"] == ["create_itinerary", "calculate_budget"]
    assert result["itinerary"]
    assert result["budget"]["total_budget"] == 4500.0
    assert all(source["source_url"] for source in result["sources"])


def test_direct_answer_uses_parsed_description_not_full_document(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "أعطني نبذة عن المنطقة الشرقية",
        EASTERN_RAG_CONTEXT,
        {"destination": "Eastern Province"},
    )
    assert result["status"] == "success"
    assert "الوجهة:" not in result["answer"]
    assert "تجمع المنطقة الشرقية" in result["answer"]


def test_empty_retrieval_blocks_direct_answer(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "ما الأماكن الثقافية؟",
        [],
        {"destination": "Eastern Province"},
    )
    assert result["status"] == "error"
    assert result["error"] == "EMPTY_RETRIEVAL"


def test_out_of_scope_hotel_request_is_rejected(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "احجز لي فندقًا في الخبر",
        EASTERN_RAG_CONTEXT,
        {"destination": "Eastern Province"},
    )
    assert result["status"] == "error"
    assert result["error"] == "OUT_OF_SCOPE"


def test_out_of_scope_live_weather_request_is_rejected(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "كيف الطقس اليوم في الدمام؟",
        EASTERN_RAG_CONTEXT,
        {"destination": "Eastern Province"},
    )
    assert result["error"] == "OUT_OF_SCOPE"


def test_live_price_question_does_not_call_budget_tool(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "كم تكلفة الدخول إلى إثراء؟",
        EASTERN_RAG_CONTEXT,
        {"destination": "Eastern Province"},
    )
    assert result["error"] == "OUT_OF_SCOPE"
    assert result["tools_used"] == []


def test_budget_allocation_is_not_mistaken_for_live_price(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "وزع ميزانية 5000 ريال على الرحلة",
        [],
        {"destination": "Eastern Province", "budget": 5000},
    )
    assert result["status"] == "success"
    assert result["tools_used"] == ["calculate_budget"]


def test_invalid_budget_returns_validation_message(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "وزع الميزانية",
        [],
        {"destination": "Eastern Province", "budget": 0},
    )
    assert result["error"] == "INVALID_BUDGET"


def test_invalid_days_returns_validation_message(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "خطط لي رحلة",
        EASTERN_RAG_CONTEXT,
        {"destination": "Eastern Province", "days": 0},
    )
    assert result["error"] == "INVALID_DAYS"


def test_arabic_destination_alias_is_supported(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "خطط لي رحلة ليومين",
        EASTERN_RAG_CONTEXT,
        {"destination": "المنطقة الشرقية", "days": 2},
    )
    assert result["status"] == "success"
    assert result["destination"] == "Eastern Province"


def test_shared_output_schema_is_exact(monkeypatch):
    _disable_gemini(monkeypatch)
    result = run_agent(
        "ما الأماكن الثقافية؟",
        EASTERN_RAG_CONTEXT,
        {"destination": "Eastern Province"},
    )
    assert set(result) == {
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


def test_budget_allocation_values():
    result = calculate_budget(4000, travelers=2, days=2)
    assert result["accommodation"] == 1600.0
    assert result["food"] == 1000.0
    assert result["transport"] == 600.0
    assert result["activities"] == 600.0
    assert result["reserve"] == 200.0


def test_budget_value_does_not_auto_trigger_budget_tool(monkeypatch):
    """A populated budget field alone must not change an itinerary request."""
    _disable_gemini(monkeypatch)
    decision = decide_tools(
        "خطط لي رحلة عائلية لمدة يومين",
        {
            "destination": "Eastern Province",
            "days": 2,
            "budget": 4000,
        },
    )
    assert decision["action"] == "itinerary"
    assert decision["tools_to_use"] == ["create_itinerary"]


def test_agent_decision_rejects_action_tool_mismatch():
    """Structured validation blocks contradictory Gemini routing output."""
    from pydantic import ValidationError
    from agent import _AgentDecision

    try:
        _AgentDecision.model_validate(
            {
                "action": "direct_answer",
                "tools_to_use": ["calculate_budget"],
                "reason": "قرار متناقض للاختبار",
            }
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Contradictory decision should be rejected")


def test_tool_step_limit_is_two():
    """The Agent has an explicit stopping limit for its two tools."""
    from agent import MAX_TOOL_STEPS

    assert MAX_TOOL_STEPS == 2
