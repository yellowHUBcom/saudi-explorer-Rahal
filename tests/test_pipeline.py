"""Integration tests for pipeline -> RAG -> Agent -> Tools.

The tests replace only external retrieval/Vision calls. Agent and Python tools
run for real, so the main integration contract is verified without ChromaDB,
Sentence Transformers, network access, or a Gemini API key.
"""

from __future__ import annotations

import pipeline


EXPECTED_KEYS = {
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


def _rag_record(
    place_name: str,
    category: str,
    description: str,
    *,
    destination: str = "Eastern Province",
    distance: float = 0.1,
) -> dict:
    text = (
        f"الوجهة: {destination}\n"
        f"اسم المكان: {place_name}\n"
        f"التصنيف: {category}\n"
        f"الوصف: {description}\n"
        "المدة المقترحة: ساعتان\n"
        "مناسب لـ: العائلات\n"
        "مستوى النشاط: منخفض"
    )
    return {
        "text": text,
        "destination": destination,
        "place_name": place_name,
        "category": category,
        "source_name": "مصدر رسمي",
        "source_section": "دليل الزائر",
        "source_url": f"https://example.com/{place_name}",
        "distance": distance,
    }


def _sample_context() -> list[dict]:
    return [
        _rag_record(
            "نبذة عن المنطقة الشرقية",
            "معلومات عامة",
            "منطقة سعودية تضم وجهات ثقافية وبحرية.",
            distance=0.05,
        ),
        _rag_record(
            "مركز إثراء",
            "ثقافي",
            "مركز ثقافي يقدم معارض وتجارب معرفية.",
            distance=0.08,
        ),
        _rag_record(
            "واجهة الخبر البحرية",
            "واجهة بحرية",
            "واجهة مناسبة للمشي والأنشطة العائلية.",
            distance=0.12,
        ),
        _rag_record(
            "جزيرة المرجان",
            "طبيعي",
            "وجهة بحرية مناسبة للاسترخاء.",
            distance=0.2,
        ),
    ]


def _assert_schema(result: dict) -> None:
    assert set(result) == EXPECTED_KEYS
    assert isinstance(result["itinerary"], list)
    assert isinstance(result["budget"], dict)
    assert isinstance(result["sources"], list)
    assert isinstance(result["warnings"], list)
    assert isinstance(result["tools_used"], list)


def test_empty_question_returns_complete_error_schema():
    result = pipeline.run_pipeline(
        {"question": "", "destination": "Riyadh"}
    )

    _assert_schema(result)
    assert result["status"] == "error"
    assert result["error"] == "EMPTY_QUESTION"


def test_unsupported_destination_is_rejected():
    result = pipeline.run_pipeline(
        {"question": "ما أبرز المعالم؟", "destination": "AlUla"}
    )

    _assert_schema(result)
    assert result["status"] == "error"
    assert result["error"] == "UNSUPPORTED_DESTINATION"


def test_arabic_destination_is_normalized(monkeypatch):
    seen: dict = {}

    def fake_retrieve(*, query, destination, top_k):
        seen.update(
            query=query,
            destination=destination,
            top_k=top_k,
        )
        return _sample_context()

    monkeypatch.setattr(pipeline, "_retrieve_context", fake_retrieve)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = pipeline.run_pipeline(
        {
            "question": "ما أبرز المعالم الثقافية؟",
            "destination": "المنطقة الشرقية",
        }
    )

    _assert_schema(result)
    assert seen["destination"] == "Eastern Province"
    assert result["destination"] == "Eastern Province"
    assert result["status"] == "success"


def test_full_rag_agent_tools_integration(monkeypatch):
    """RAG evidence reaches Agent; both real Python tools execute."""

    monkeypatch.setattr(
        pipeline,
        "_retrieve_context",
        lambda **_: _sample_context(),
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = pipeline.run_pipeline(
        {
            "question": (
                "خطط لي رحلة عائلية لمدة يومين في المنطقة الشرقية "
                "ووزع ميزانية 4000 ريال"
            ),
            "destination": "Eastern Province",
            "days": 2,
            "travelers": 2,
            "budget": 4000,
            "interests": ["ثقافي", "واجهة بحرية"],
        }
    )

    _assert_schema(result)
    assert result["status"] == "success"
    assert result["error"] is None
    assert result["tools_used"] == [
        "create_itinerary",
        "calculate_budget",
    ]
    assert len(result["itinerary"]) == 2
    assert result["budget"]["total_budget"] == 4000
    assert result["budget"]["accommodation"] == 1600
    assert result["budget"]["food"] == 1000
    assert result["budget"]["transport"] == 600
    assert result["budget"]["activities"] == 600
    assert result["budget"]["reserve"] == 200
    assert result["sources"]

    itinerary_places = {
        activity["place_name"]
        for day in result["itinerary"]
        for activity in day["activities"]
    }
    assert "نبذة عن المنطقة الشرقية" not in itinerary_places
    assert itinerary_places <= {
        "مركز إثراء",
        "واجهة الخبر البحرية",
        "جزيرة المرجان",
    }


def test_direct_question_uses_rag_without_tools(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_retrieve_context",
        lambda **_: _sample_context(),
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = pipeline.run_pipeline(
        {
            "question": "ما أبرز المعالم الثقافية في المنطقة الشرقية؟",
            "destination": "Eastern Province",
        }
    )

    _assert_schema(result)
    assert result["status"] == "success"
    assert result["tools_used"] == []
    assert result["sources"]


def test_budget_only_works_when_rag_is_empty(monkeypatch):
    monkeypatch.setattr(pipeline, "_retrieve_context", lambda **_: [])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = pipeline.run_pipeline(
        {
            "question": "وزع ميزانية 2000 ريال",
            "destination": "Jeddah",
            "budget": 2000,
            "travelers": 1,
        }
    )

    _assert_schema(result)
    assert result["status"] == "success"
    assert result["tools_used"] == ["calculate_budget"]
    assert result["budget"]["total_budget"] == 2000


def test_itinerary_is_blocked_when_rag_is_empty(monkeypatch):
    monkeypatch.setattr(pipeline, "_retrieve_context", lambda **_: [])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = pipeline.run_pipeline(
        {
            "question": "خطط لي رحلة لمدة يومين",
            "destination": "Abha",
            "days": 2,
        }
    )

    _assert_schema(result)
    assert result["status"] == "error"
    assert result["error"] == "EMPTY_RETRIEVAL"
    assert result["tools_used"] == []


def test_invalid_optional_values_are_rejected_before_rag(monkeypatch):
    called = False

    def fake_retrieve(**_):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(pipeline, "_retrieve_context", fake_retrieve)

    result = pipeline.run_pipeline(
        {
            "question": "خطط لي رحلة",
            "destination": "Riyadh",
            "days": 0,
        }
    )

    _assert_schema(result)
    assert result["error"] == "INVALID_DAYS"
    assert called is False


def test_rag_failure_returns_clear_error(monkeypatch):
    def raise_rag(**_):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(pipeline, "_retrieve_context", raise_rag)

    result = pipeline.run_pipeline(
        {
            "question": "ما أبرز المعالم؟",
            "destination": "Riyadh",
        }
    )

    _assert_schema(result)
    assert result["status"] == "error"
    assert result["error"] == "RAG_FAILURE"


def test_supported_image_can_supply_destination(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_identify_landmark",
        lambda _image: {
            "status": "supported",
            "destination": "Eastern Province",
            "landmark": "Ithra",
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_retrieve_context",
        lambda **_: _sample_context(),
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = pipeline.run_pipeline(
        {
            "question": "ما هذا المعلم؟",
            "destination": None,
            "image": object(),
        }
    )

    _assert_schema(result)
    assert result["status"] == "success"
    assert result["destination"] == "Eastern Province"
