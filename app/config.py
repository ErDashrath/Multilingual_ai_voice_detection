"""Configuration settings for the AI Voice Detection API"""

import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""
    
    # API Configuration
    API_KEY: str = os.getenv("API_KEY", "sk_test_123456789")
    
    # Model Settings
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./models_cache")
    
    # Audio Settings
    MAX_AUDIO_DURATION_SECONDS: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "300"))
    SAMPLE_RATE: int = 16000  # Required by models
    
    # Supported Languages
    ALLOWED_LANGUAGES: List[str] = [
        lang.strip()
        for lang in os.getenv(
            "ALLOWED_LANGUAGES",
            "Tamil,English,Hindi,Malayalam,Telugu"
        ).split(",")
        if lang.strip()
    ]
    
    # Language ISO Code Mapping
    LANGUAGE_ISO_MAP: dict = {
        "ta": "Tamil",
        "en": "English",
        "hi": "Hindi",
        "ml": "Malayalam",
        "te": "Telugu"
    }
    
    # Reverse mapping
    ISO_LANGUAGE_MAP: dict = {v: k for k, v in LANGUAGE_ISO_MAP.items()}
    
    # Debug Mode
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Names
    DEEPFAKE_MODEL: str = "MelodyMachine/Deepfake-audio-detection-V2"
    LANGUAGE_MODEL: str = "speechbrain/lang-id-voxlingua107-ecapa"


settings = Settings()
