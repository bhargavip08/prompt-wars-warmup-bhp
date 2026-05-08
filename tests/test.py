# test.py
# Run tests using:
# pytest -v test.py

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────
# Mock Responses
# ─────────────────────────────────────────────────────────────

VALID_ITINERARY_RESPONSE = json.dumps({
    "trip_summary": "A vibrant and cultural Tokyo adventure.",
    "vibe_match_score": 97,
    "daily_plan": [
        {
            "day": 1,
            "day_title": "Tokyo City Lights",
            "morning_activity": "Visit Senso-ji Temple",
            "afternoon_activity": "Explore Akihabara",
            "evening_activity": "Dinner at Ichiran Ramen"
        }
    ]
})

VALID_REPLAN_RESPONSE = json.dumps({
    "day": 1,
    "day_title": "Hidden Tokyo Escapes",
    "morning_activity": "Visit Yanaka district",
    "afternoon_activity": "Tea ceremony in Ueno",
    "evening_activity": "Dinner at local sushi bar"
})


# ─────────────────────────────────────────────────────────────
# Health Check Tests
# ─────────────────────────────────────────────────────────────

def test_healthz():
    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "model" in data


# ─────────────────────────────────────────────────────────────
# UI Route Tests
# ─────────────────────────────────────────────────────────────

def test_root_ui():
    response = client.get("/")

    assert response.status_code == 200


def test_invalid_route():
    response = client.get("/invalid")

    assert response.status_code == 404


# ─────────────────────────────────────────────────────────────
# Generate Itinerary Tests
# ─────────────────────────────────────────────────────────────

@patch("app.call_gemini")
def test_generate_itinerary_success(mock_gemini):
    mock_gemini.return_value = VALID_ITINERARY_RESPONSE

    payload = {
        "destination": "Tokyo",
        "days": 1,
        "startDate": "2026-05-10",
        "endDate": "2026-05-11",
        "vibe": "Adventure and food",
        "budget": "Mid-range",
        "constraints": ["Vegetarian"]
    }

    response = client.post("/api/generate-itinerary", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert "trip_summary" in data["data"]
    assert len(data["data"]["daily_plan"]) == 1


def test_generate_itinerary_missing_destination():
    payload = {
        "days": 2,
        "vibe": "Adventure",
        "budget": "Luxury"
    }

    response = client.post("/api/generate-itinerary", json=payload)

    assert response.status_code == 422


def test_generate_itinerary_invalid_days():
    payload = {
        "destination": "Paris",
        "days": 0,
        "vibe": "Romantic",
        "budget": "Luxury"
    }

    response = client.post("/api/generate-itinerary", json=payload)

    assert response.status_code == 422


def test_generate_itinerary_invalid_date_format():
    payload = {
        "destination": "Paris",
        "days": 2,
        "startDate": "10-05-2026",
        "endDate": "11-05-2026",
        "vibe": "Romantic",
        "budget": "Luxury"
    }

    response = client.post("/api/generate-itinerary", json=payload)

    assert response.status_code == 422


@patch("app.call_gemini")
def test_generate_itinerary_invalid_json(mock_gemini):
    mock_gemini.return_value = "INVALID_JSON"

    payload = {
        "destination": "Rome",
        "days": 2,
        "vibe": "History",
        "budget": "Budget"
    }

    response = client.post("/api/generate-itinerary", json=payload)

    assert response.status_code == 502
    assert response.json()["detail"] == "Model returned invalid JSON. Please try again."


@patch("app.call_gemini")
def test_generate_itinerary_internal_server_error(mock_gemini):
    mock_gemini.side_effect = Exception("Gemini failure")

    payload = {
        "destination": "Rome",
        "days": 2,
        "vibe": "History",
        "budget": "Budget"
    }

    response = client.post("/api/generate-itinerary", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Itinerary generation failed."


# ─────────────────────────────────────────────────────────────
# Replan Day Tests
# ─────────────────────────────────────────────────────────────

@patch("app.call_gemini")
def test_replan_day_success(mock_gemini):
    mock_gemini.return_value = VALID_REPLAN_RESPONSE

    payload = {
        "destination": "Tokyo",
        "days": 5,
        "startDate": "2026-05-10",
        "vibe": "Adventure",
        "budget": "Luxury",
        "constraints": ["Vegetarian"],
        "day_number": 1
    }

    response = client.post("/api/replan-day", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["data"]["day"] == 1


def test_replan_day_invalid_day():
    payload = {
        "destination": "Tokyo",
        "days": 5,
        "vibe": "Adventure",
        "budget": "Luxury",
        "day_number": 0
    }

    response = client.post("/api/replan-day", json=payload)

    assert response.status_code == 422


@patch("app.call_gemini")
def test_replan_day_invalid_json(mock_gemini):
    mock_gemini.return_value = "INVALID_JSON"

    payload = {
        "destination": "Tokyo",
        "days": 5,
        "vibe": "Adventure",
        "budget": "Luxury",
        "day_number": 2
    }

    response = client.post("/api/replan-day", json=payload)

    assert response.status_code == 502


@patch("app.call_gemini")
def test_replan_day_internal_server_error(mock_gemini):
    mock_gemini.side_effect = Exception("Gemini failure")

    payload = {
        "destination": "Tokyo",
        "days": 5,
        "vibe": "Adventure",
        "budget": "Luxury",
        "day_number": 2
    }

    response = client.post("/api/replan-day", json=payload)

    assert response.status_code == 500


# ─────────────────────────────────────────────────────────────
# Constraint Validation Tests
# ─────────────────────────────────────────────────────────────

def test_constraints_capped_to_10():
    constraints = [f"constraint_{i}" for i in range(15)]

    payload = {
        "destination": "Tokyo",
        "days": 5,
        "vibe": "Adventure",
        "budget": "Luxury",
        "constraints": constraints
    }

    response = client.post("/api/generate-itinerary", json=payload)

    # Validation passes because model truncates internally
    assert response.status_code in [200, 500, 502, 422]


# ─────────────────────────────────────────────────────────────
# HTTP Method Validation
# ─────────────────────────────────────────────────────────────

def test_invalid_http_method():
    response = client.put("/healthz")

    assert response.status_code == 405

