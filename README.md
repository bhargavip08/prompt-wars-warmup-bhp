# 🌍 AI Travel Experience Engine (BHP Warmup)

A modern, fast, and reliable travel itinerary generator powered by **FastAPI** and **Google Gemini 3**. This application transforms user preferences (destination, duration, vibe, budget) into a structured, day-wise travel plan.

## 🚀 Features
- **Dynamic Model Discovery**: Automatically detects and connects to the best available Gemini model (Gemini 3 Flash / 1.5 Flash).
- **Strict JSON Mode**: Leverages Gemini's `response_mime_type` for guaranteed data structure.
- **Async Architecture**: Built with FastAPI for non-blocking, high-performance API calls.
- **Modern UI**: Clean, responsive interface using Jinja2 templates.

## 🛠️ Tech Stack
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **AI**: Google Generative AI SDK (Gemini 3 Flash)
- **Frontend**: HTML5, Tailwind CSS (via CDN), JavaScript (Fetch API)
- **Environment**: Python-dotenv for secure API key management

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd prompt-wars-warmup-bhp