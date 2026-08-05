from typing import Any
from agent import run_agent
from rag import retrieve_context
from vision import identify_landmark

def run_pipeline(input_payload: dict[str, Any]) -> dict[str, Any]:
    question = str(input_payload.get("question") or "").strip()
    destination = input_payload.get("destination")
    image = input_payload.get("image")

    # التحقق من وجود سؤال
    if not question:
        return {
            "status": "error",
            "answer": "يرجى كتابة سؤال متعلق بالرحلة.",
            "error": "EMPTY_QUESTION"
        }

    warnings: list[str] = []

    # معالجة الصورة عبر Vision إن وجدت
    if image is not None:
        vision_result = identify_landmark(image)
        if vision_result.get("status") == "supported":
            destination = vision_result.get("destination")
        elif not destination:
            return {
                "status": "error",
                "answer": "تعذر تحديد المعلم؛ يرجى اختيار الوجهة يدويًا.",
                "error": "MANUAL_DESTINATION_REQUIRED"
            }
        else:
            warnings.append("تعذر التعرف على الصورة تلقائيًا، تم استخدام الوجهة المحددة يدويًا.")

    # التأكد من تحديد وجهة
    if not destination:
        return {
            "status": "error",
            "answer": "يرجى اختيار وجهة أو رفع صورة معلم مدعوم.",
            "error": "MISSING_DESTINATION"
        }

    # استرجاع السياق عبر RAG
    context = retrieve_context(question, destination, top_k=4)

    # تنفيذ منطق الـ Agent
    result = run_agent(
        user_query=question,
        context=context,
        preferences={
            "destination": destination,
            "days": input_payload.get("days"),
            "travelers": input_payload.get("travelers"),
            "budget": input_payload.get("budget"),
            "interests": input_payload.get("interests") or [],
        },
    )

    result["warnings"] = warnings + result.get("warnings", [])
    return result
