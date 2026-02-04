"""
AI Voice Detection API - Main Application

Detects whether a voice recording is AI-generated or Human
across Tamil, English, Hindi, Malayalam, and Telugu languages.

Models Used:
- Deepfake Detection: MelodyMachine/Deepfake-audio-detection-V2 (99.73% accuracy)
- Language ID: speechbrain/lang-id-voxlingua107-ecapa (93.3% accuracy)
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
import os

from .routes import voice_detection_router
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler
    - Startup: Fast startup (lazy model loading)
    - Shutdown: Cleanup
    """
    # Startup - Fast, no model preloading
    print("=" * 60)
    print("🚀 AI Voice Detection API Starting...")
    print("=" * 60)
    print(f"📦 Deepfake Model: {settings.DEEPFAKE_MODEL}")
    print(f"🌍 Language Model: {settings.LANGUAGE_MODEL}")
    print(f"🗣️  Supported Languages: {', '.join(settings.ALLOWED_LANGUAGES)}")
    print("=" * 60)
    print("⚡ Lightweight mode: Models load on first request")
    print("✅ API Ready! Listening for requests...")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("👋 API Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="AI Voice Detection API",
    description="""
## 🎙️ AI Voice Detection API

Detects whether a voice recording is **AI-generated** or **Human**.

### Supported Languages
- 🇮🇳 Tamil
- 🇬🇧 English  
- 🇮🇳 Hindi
- 🇮🇳 Malayalam
- 🇮🇳 Telugu

### Models Used
- **Deepfake Detection:** MelodyMachine/Deepfake-audio-detection-V2 (99.73% accuracy)
- **Language ID:** speechbrain/lang-id-voxlingua107-ecapa

### Authentication
All requests require an API key via `x-api-key` header.

### Input Format
- Audio: MP3 format, Base64 encoded
- One audio file per request
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


# HTTP exception handler to normalize error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return errors in the documented ErrorResponse shape"""
    if isinstance(exc.detail, dict) and "status" in exc.detail and "message" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": str(exc.detail)
        }
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error. Please try again later."
        }
    )


# Include routers
app.include_router(voice_detection_router)

# Get static directory path
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


# Serve frontend at root
@app.get("/", tags=["Frontend"], response_class=HTMLResponse)
async def serve_frontend():
    """Serve the test frontend"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


# API info endpoint
@app.get("/api", tags=["Root"])
async def api_info():
    """API information endpoint"""
    return {
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "description": "Detects AI-generated vs Human voice",
        "documentation": "/docs",
        "health_check": "/health",
        "supported_languages": settings.ALLOWED_LANGUAGES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
