"""Voice Detection API Routes"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Union

from ..models.schemas import (
    VoiceDetectionRequest,
    VoiceDetectionResponse,
    ErrorResponse
)
from ..middleware.auth import verify_api_key
from ..services import (
    AudioProcessor,
    LanguageDetector,
    DeepfakeDetector,
    ExplanationGenerator
)
from ..config import settings


router = APIRouter(tags=["Voice Detection"])

# Initialize services (singleton pattern - loaded once)
audio_processor = AudioProcessor()
explanation_generator = ExplanationGenerator()

# Models are loaded lazily on first request
_language_detector = None
_deepfake_detector = None


def get_language_detector() -> LanguageDetector:
    """Get or create language detector instance"""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    return _language_detector


def get_deepfake_detector() -> DeepfakeDetector:
    """Get or create deepfake detector instance"""
    global _deepfake_detector
    if _deepfake_detector is None:
        _deepfake_detector = DeepfakeDetector()
    return _deepfake_detector


@router.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        200: {"model": VoiceDetectionResponse, "description": "Successful detection"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Detect AI-Generated or Human Voice",
    description="""
    Analyzes an audio file to determine if the voice is AI-generated or spoken by a human.
    
    **Supported Languages:** Tamil, English, Hindi, Malayalam, Telugu
    
    **Audio Format:** MP3 (Base64 encoded)
    
    **Authentication:** API Key required via `x-api-key` header
    """
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
) -> VoiceDetectionResponse:
    """
    Main voice detection endpoint
    
    Process:
    1. Validate API key
    2. Decode Base64 audio
    3. Convert to 16kHz WAV
    4. Extract audio features
    5. Run AI/Human detection
    6. Generate explanation
    7. Return results
    """
    try:
        # Step 1: Validate language is supported
        if request.language not in settings.ALLOWED_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": f"Unsupported language: {request.language}. Supported: {', '.join(settings.ALLOWED_LANGUAGES)}"
                }
            )
        
        # Step 2: Process audio (decode, convert, validate, extract features)
        try:
            audio_array, audio_features = audio_processor.process(request.audioBase64)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": str(e)
                }
            )
        
        # Step 3: Run deepfake detection
        try:
            deepfake_detector = get_deepfake_detector()
            detection_result = deepfake_detector.detect(audio_array)
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "status": "error",
                    "message": f"Detection model error: {str(e)}"
                }
            )
        
        # Step 4: Generate explanation based on audio features and detection
        explanation = explanation_generator.generate(
            classification=detection_result.classification,
            confidence=detection_result.confidence,
            audio_features=audio_features
        )
        
        # Step 5: Build and return response
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=detection_result.classification,
            confidenceScore=detection_result.confidence,
            explanation=explanation
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )


@router.get(
    "/health",
    summary="Health Check",
    description="Check if the API is running"
)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Voice Detection API",
        "version": "1.0.0"
    }


@router.get(
    "/api/supported-languages",
    summary="Get Supported Languages",
    description="Returns list of supported languages for voice detection"
)
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "status": "success",
        "languages": settings.ALLOWED_LANGUAGES,
        "count": len(settings.ALLOWED_LANGUAGES)
    }
