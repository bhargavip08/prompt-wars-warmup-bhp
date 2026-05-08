import os
import json
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai

# Cloud Run automatically provides this from Secret Manager now
api_key = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

def get_bulletproof_model():
    """Dynamically finds an available model to avoid latency"""
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Check for Gemini 3 first, then 1.5 variants
        for target in ['models/gemini-3-flash', 'models/gemini-1.5-flash', 'models/gemini-1.5-flash-latest']:
            if target in available:
                print(f"✅ AI linked successfully using: {target}")
                return target
        return available[0] if available else 'models/gemini-1.5-flash'
    except Exception:
        return 'models/gemini-1.5-flash'

SELECTED_MODEL = get_bulletproof_model()

# --- 2. APP INITIALIZATION ---
app = FastAPI(title="Travel Engine API")
templates = Jinja2Templates(directory="templates")

# --- 3. MODELS ---
class TravelRequest(BaseModel):
    destination: str
    days: int
    vibe: str
    budget: str

# --- 4. ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    # Standard keyword argument syntax for modern FastAPI
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={}
    )

@app.post("/api/generate-itinerary")
async def generate_itinerary(req: TravelRequest):
    try:
        # Re-initialize with current selected model and JSON config
        model = genai.GenerativeModel(
            model_name=SELECTED_MODEL,
            generation_config={
                "temperature": 0.7,
                "response_mime_type": "application/json",
            }
        )
        
        prompt = f"""
        You are a dynamic travel planning engine. 
        User wants to visit {req.destination} for {req.days} days.
        Vibe: {req.vibe}, Budget: {req.budget}.
        
        Return a JSON object with:
        'trip_summary': A brief overview string.
        'daily_plan': An array of objects with 'day', 'morning_activity', 'afternoon_activity', and 'evening_activity'.
        """
        
        response = model.generate_content(prompt)
        
        # Parse the JSON string returned by Gemini
        itinerary = json.loads(response.text)
        return {"status": "success", "data": itinerary}
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run locally using: python -m uvicorn app:app --reload --port 8080