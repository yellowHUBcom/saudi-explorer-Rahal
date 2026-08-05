from typing import Any
from agent import run_agent
from rag import retrieve_context
from vision import identify_landmark

def run_pipeline(input_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Executes the main application pipeline by orchestrating Vision, RAG, 
    and Agent operations based on user input.

    Args:
        input_payload (dict[str, Any]): Dictionary containing user inputs including
                                        'question', 'destination', 'image', 'days',
                                        'travelers', 'budget', and 'interests'.

    Returns:
        dict[str, Any]: Structured dictionary compliant with the Shared Output Schema.
    """
    question = str(input_payload.get("question") or "").strip()
    destination = input_payload.get("destination")
    image = input_payload.get("image")

    # Validate mandatory text input
    if not question:
        return {
            "status": "error",
            "answer": "يرجى كتابة سؤال متعلق بالرحلة.",
            "error": "EMPTY_QUESTION"
        }

    warnings: list[str] = []

    # Process input image via Vision model if provided
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

    # Ensure a target destination is specified
    if not destination:
        return {
            "status": "error",
            "answer": "يرجى اختيار وجهة أو رفع صورة معلم مدعوم.",
            "error": "MISSING_DESTINATION"
        }

    # Retrieve domain knowledge context using RAG module
    context = retrieve_context(question, destination, top_k=4)

    # Execute core reasoning agent with contextual constraints and user preferences
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

    # Merge internal processing warnings with agent execution warnings
    result["warnings"] = warnings + result.get("warnings", [])
    return result
