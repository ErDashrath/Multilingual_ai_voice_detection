"""API Integration Tests"""

import pytest
from fastapi.testclient import TestClient
import base64
import os

from app.main import app
from app.config import settings


client = TestClient(app)

# Valid Base64 payload for tests (not a real MP3, but passes schema validation)
VALID_AUDIO_BASE64 = base64.b64encode(b"\x00" * 200).decode("utf-8")


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test /health endpoint returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root_endpoint(self):
        """Test /api endpoint returns API info"""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AI Voice Detection API"
        assert "supported_languages" in data


class TestAuthentication:
    """Test API key authentication"""
    
    def test_missing_api_key(self):
        """Test request without API key is rejected"""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": VALID_AUDIO_BASE64
            }
        )
        assert response.status_code == 401
        data = response.json()
        assert data["status"] == "error"
    
    def test_invalid_api_key(self):
        """Test request with invalid API key is rejected"""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": VALID_AUDIO_BASE64
            },
            headers={"x-api-key": "invalid_key"}
        )
        assert response.status_code == 401


class TestRequestValidation:
    """Test request body validation"""
    
    def test_invalid_language(self):
        """Test unsupported language is rejected"""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "French",
                "audioFormat": "mp3",
                "audioBase64": VALID_AUDIO_BASE64
            },
            headers={"x-api-key": settings.API_KEY}
        )
        assert response.status_code == 422  # Validation error
    
    def test_invalid_audio_format(self):
        """Test unsupported audio format is rejected"""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "wav",
                "audioBase64": VALID_AUDIO_BASE64
            },
            headers={"x-api-key": settings.API_KEY}
        )
        assert response.status_code == 422


class TestSupportedLanguages:
    """Test supported languages endpoint"""
    
    def test_get_supported_languages(self):
        """Test /api/supported-languages returns all languages"""
        response = client.get("/api/supported-languages")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Tamil" in data["languages"]
        assert "English" in data["languages"]
        assert "Hindi" in data["languages"]
        assert "Malayalam" in data["languages"]
        assert "Telugu" in data["languages"]
        assert data["count"] == 5


# Integration test with actual audio (requires audio file)
class TestVoiceDetection:
    """Test voice detection with actual audio"""
    
    @pytest.mark.skipif(
        not os.path.exists("tests/test_audio/sample.mp3"),
        reason="Test audio file not found"
    )
    def test_voice_detection_with_audio(self):
        """Test full voice detection pipeline with real audio"""
        # Load test audio
        with open("tests/test_audio/sample.mp3", "rb") as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            },
            headers={"x-api-key": settings.API_KEY}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["language"] == "English"
        assert data["classification"] in ["AI_GENERATED", "HUMAN"]
        assert 0 <= data["confidenceScore"] <= 1
        assert len(data["explanation"]) > 0
