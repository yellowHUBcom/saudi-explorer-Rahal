import os
import json
import google.generativeai as genai
from PIL import Image

def identify_landmark(image: Image.Image) -> dict:
    """
    Identifies Saudi landmarks from an uploaded image using Gemini Vision.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "status": "unsupported",
            "destination": None,
            "landmark": None,
            "error": "Gemini API key is missing."
        }
    
    genai.configure(api_key=api_key)
    
    prompt = """
    Analyze this image. Is it one of the following supported landmarks in Saudi Arabia?
    - Kingdom Centre (Riyadh)
    - King Fahd Fountain (Jeddah)
    - Elephant Rock (AlUla)
    - Green Mountain (Abha)
    - Ithra / King Abdulaziz Center (Eastern Province)

    Reply ONLY with a valid raw JSON object (no markdown, no ```json wrapper) matching this exact format:
    {
      "status": "supported" | "unsupported",
      "destination": "Riyadh" | "Jeddah" | "AlUla" | "Abha" | "Eastern Province" | null,
      "landmark": "Name of landmark" | null
    }
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, image])
        
        # Clean response in case model returns code blocks
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data
    except Exception as e:
        return {
            "status": "unsupported", 
            "destination": None, 
            "landmark": None,
            "error": str(e)
        }
