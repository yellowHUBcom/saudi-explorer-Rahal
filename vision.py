import json
import os

import google.generativeai as genai
from PIL import Image


def identify_landmark(image: Image.Image) -> dict:
    """Identify supported Saudi landmarks using Gemini Vision."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "status": "unsupported",
            "destination": None,
            "landmark": None,
            "error": "Gemini API key is missing.",
        }

    genai.configure(api_key=api_key)

    prompt = """
Analyze this image and determine whether it matches one of the supported
Saudi landmarks listed below.

Supported landmarks and required output names:
- Kingdom Centre, Riyadh → برج المملكة
- King Fahd Fountain, Jeddah → نافورة الملك فهد
- Green Mountain, Abha → الجبل الأخضر
- Ithra / King Abdulaziz Center, Eastern Province → مركز إثراء

Important rules:
- Choose only from the supported landmarks above.
- Return the landmark name in Arabic exactly as written above.
- Return the destination in English exactly as one of:
  Riyadh, Jeddah, Abha, Eastern Province
- If the image does not clearly match a supported landmark, return unsupported.
- Reply only with valid raw JSON.
- Do not include Markdown or code fences.

Use exactly this structure:
{
  "status": "supported" or "unsupported",
  "destination": "Riyadh" or "Jeddah" or "Abha" or "Eastern Province" or null,
  "landmark": "برج المملكة" or "نافورة الملك فهد" or
              "الجبل الأخضر" or "مركز إثراء" or null
}
"""

    try:
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content([prompt, image])

        clean_text = (
            response.text
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

        data = json.loads(clean_text)
        print("VISION RESPONSE:", data)
        return data

    except Exception as error:
        print("VISION ERROR:", repr(error))

        return {
            "status": "unsupported",
            "destination": None,
            "landmark": None,
            "error": str(error),
        }
   