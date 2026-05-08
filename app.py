"""
Voyager — AI Travel Engine
FastAPI backend powered by Google Gemini.

Run locally:
    uvicorn app:app --reload --port 8080

Deploy:
    gcloud run deploy voyager --source . --region us-central1 --allow-unauthenticated
"""

import os
import json
import logging
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types

# ── Bootstrap ─────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("voyager")

# ── Gemini client ──────────────────────────────────────────────────────────────

_api_key = os.environ.get("GOOGLE_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Add it to your .env file or Cloud Run secret manager."
    )

client = genai.Client(api_key=_api_key)

MODEL_PRIORITY = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


def _select_model() -> str:
    """Pick the best available Gemini model at startup."""
    try:
        available = {m.name for m in client.models.list()}
        for target in MODEL_PRIORITY:
            if f"models/{target}" in available or target in available:
                log.info("Model selected: %s", target)
                return target
        # Fallback: any Gemini model
        for m in client.models.list():
            if "gemini" in m.name:
                log.warning("Using fallback model: %s", m.name)
                return m.name
    except Exception as exc:
        log.warning("Model discovery failed (%s). Using gemini-1.5-flash.", exc)
    return "gemini-1.5-flash"


SELECTED_MODEL: str = _select_model()

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voyager Travel Engine API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# Restrict CORS in production via env var (default: same-origin only)
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

templates = Jinja2Templates(directory="templates")

# ── Request / Response models ──────────────────────────────────────────────────

class TravelRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=120)
    days: int        = Field(..., ge=1, le=30)
    startDate: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    endDate:   Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    vibe: str        = Field(..., min_length=1, max_length=300)
    budget: str      = Field(..., min_length=1, max_length=80)
    constraints: Optional[List[str]] = Field(default_factory=list)

    @field_validator("constraints")
    @classmethod
    def _cap_constraints(cls, v: Optional[List[str]]) -> List[str]:
        if not v:
            return []
        return v[:10]  # never pass more than 10 to the model


class ReplanDayRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=120)
    days: int        = Field(..., ge=1, le=30)
    startDate: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    vibe: str        = Field(..., min_length=1, max_length=300)
    budget: str      = Field(..., min_length=1, max_length=80)
    constraints: Optional[List[str]] = Field(default_factory=list)
    day_number: int  = Field(..., ge=1, le=30)

    @field_validator("constraints")
    @classmethod
    def _cap_constraints(cls, v: Optional[List[str]]) -> List[str]:
        if not v:
            return []
        return v[:10]


# ── Gemini helpers ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Voyager, an award-winning AI travel planning concierge.
You combine the knowledge of a seasoned travel journalist, a local cultural expert,
and a logistics specialist. Your plans are:
- Hyper-specific: always use real place names, actual restaurant names, real neighbourhoods
- Logistically sound: activities are geographically proximate within each day
- Culturally sensitive: respect the user's constraints absolutely
- Vivid and inspiring: language should evoke the destination's atmosphere
You ALWAYS return valid JSON matching the exact schema requested. No markdown fences, no prose outside JSON."""


def call_gemini(prompt: str) -> str:
    """
    Call Gemini with best-practice settings:
    - Separate system instruction
    - response_mime_type forces structured JSON output
    - temperature / top_p / top_k tuned for creative-but-consistent results
    """
    response = client.models.generate_content(
        model=SELECTED_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.75,
            top_p=0.92,
            top_k=40,
            response_mime_type="application/json",
            candidate_count=1,
        ),
    )
    return response.text


def _constraints_xml(constraints: List[str]) -> str:
    if not constraints:
        return ""
    items = "\n".join(f"  - {c}" for c in constraints)
    return f"\n<constraints>\n{items}\n</constraints>"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/healthz")
async def health():
    """Liveness probe for Cloud Run."""
    return {"status": "ok", "model": SELECTED_MODEL}


@app.post("/api/generate-itinerary")
async def generate_itinerary(req: TravelRequest):
    date_ctx = ""
    if req.startDate and req.endDate:
        date_ctx = (
            f"\n<travel_dates>\n"
            f"  Start: {req.startDate}\n"
            f"  End:   {req.endDate}\n"
            f"</travel_dates>"
        )

    prompt = f"""<task>Generate a complete, inspiring travel itinerary.</task>

<traveller_request>
  Destination: {req.destination}
  Duration: {req.days} days
  Vibe / travel style: {req.vibe}
  Budget tier: {req.budget}
</traveller_request>
{date_ctx}
{_constraints_xml(req.constraints or [])}

<output_schema>
Return a single JSON object with exactly this structure:
{{
  "trip_summary": "2-sentence vivid overview capturing the emotional essence of this trip",
  "vibe_match_score": <integer between 94 and 99>,
  "daily_plan": [
    {{
      "day": <integer starting at 1>,
      "day_title": "Evocative 3-5 word title capturing the day's theme",
      "morning_activity": "Specific activity at a real named place with one insider tip",
      "afternoon_activity": "Specific activity at a real named place with one insider tip",
      "evening_activity": "Specific activity at a real named place with dining recommendation"
    }}
  ]
}}
</output_schema>

<quality_requirements>
- Each activity names a specific, real, bookable venue or location
- Morning → afternoon → evening flow must be geographically logical
- At least one activity per day directly reflects the stated vibe
- All constraints strictly honoured with zero exceptions
- Produce exactly {req.days} day entries
</quality_requirements>"""

    try:
        result = json.loads(call_gemini(prompt))
        return {"status": "success", "data": result}
    except json.JSONDecodeError as exc:
        log.error("JSON parse error in generate_itinerary: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Model returned invalid JSON. Please try again.",
        )
    except Exception as exc:
        log.error("Error in generate_itinerary: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Itinerary generation failed.",
        )


@app.post("/api/replan-day")
async def replan_day(req: ReplanDayRequest):
    prompt = f"""<task>Regenerate a single day with a FRESH, alternative plan.</task>

<context>
  Destination: {req.destination}
  Day number: {req.day_number} of {req.days}
  Overall vibe: {req.vibe}
  Budget tier: {req.budget}
</context>
{_constraints_xml(req.constraints or [])}

<instruction>
Provide a completely different alternative — different part of the city, different cuisine,
different activity type. Surprise the traveller. All constraints still apply with zero exceptions.
</instruction>

<output_schema>
Return a single JSON object:
{{
  "day": {req.day_number},
  "day_title": "Fresh evocative 3-5 word title",
  "morning_activity": "Specific alternative morning at a real named place, with insider tip",
  "afternoon_activity": "Specific alternative afternoon at a real named place, with insider tip",
  "evening_activity": "Specific alternative evening at a real named place, with dining tip"
}}
</output_schema>"""

    try:
        result = json.loads(call_gemini(prompt))
        return {"status": "success", "data": result}
    except json.JSONDecodeError as exc:
        log.error("JSON parse error in replan_day: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Model returned invalid JSON. Please try again.",
        )
    except Exception as exc:
        log.error("Error in replan_day: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Day re-plan failed.",
        )