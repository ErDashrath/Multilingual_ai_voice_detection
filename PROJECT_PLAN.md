# 🎯 AI Voice Detection API - Project Plan

## 📋 Project Overview
Build a secure REST API that detects whether a voice recording is **AI-generated** or **Human** across 5 Indian languages: Tamil, English, Hindi, Malayalam, Telugu.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           CLIENT REQUEST                                  │
│                    (Base64 MP3 + Language + API Key)                     │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI SERVER                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │ API Key     │───▶│ Audio       │───▶│ Language    │───▶│ AI/Human  │ │
│  │ Validation  │    │ Processing  │    │ Detection   │    │ Detection │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           JSON RESPONSE                                   │
│     (status, language, classification, confidenceScore, explanation)     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

# 📅 PHASE 1: Project Setup & Foundation
**Duration:** Day 1 | **Priority:** Critical

## F1.1 - Project Structure Creation
**Description:** Initialize the project with proper folder structure and configuration files.

**Tasks:**
- [ ] Create main project directory structure
- [ ] Initialize Python virtual environment
- [ ] Create `requirements.txt` with all dependencies
- [ ] Create `.env` file for environment variables
- [ ] Create `.gitignore` file

**Folder Structure:**
```
aivoice_detection/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration settings
│   ├── routes/
│   │   ├── __init__.py
│   │   └── voice_detection.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_processor.py
│   │   ├── language_detector.py
│   │   └── deepfake_detector.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── auth.py          # API Key validation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_audio/          # Sample test files
├── models_cache/            # Downloaded ML models
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

**Output:** Complete project skeleton ready for development

---

## F1.2 - Dependencies Installation
**Description:** Install all required Python packages.

**Dependencies:**
```txt
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Audio Processing
librosa==0.10.1
pydub==0.25.1
soundfile==0.12.1
ffmpeg-python==0.2.0

# ML Models
torch==2.1.2
torchaudio==2.1.2
transformers==4.41.2
speechbrain==1.0.0

# Utilities
python-dotenv==1.0.0
pydantic==2.5.3
numpy==1.26.3

# Testing
pytest==7.4.4
httpx==0.26.0
```

**Output:** All packages installed and verified

---

## F1.3 - Configuration Setup
**Description:** Set up environment variables and configuration management.

**Environment Variables (.env):**
```env
API_KEY=sk_test_your_secret_key_here
MODEL_CACHE_DIR=./models_cache
MAX_AUDIO_DURATION_SECONDS=300
ALLOWED_LANGUAGES=Tamil,English,Hindi,Malayalam,Telugu
DEBUG=False
```

**Output:** Configuration system ready

---

# 📅 PHASE 2: Audio Processing Pipeline
**Duration:** Day 1-2 | **Priority:** Critical

## F2.1 - Base64 Decoder
**Description:** Decode Base64 encoded MP3 audio from request body.

**Input:** Base64 string
**Output:** Raw MP3 bytes

**Implementation Details:**
- Validate Base64 format
- Handle padding issues
- Return decoded bytes
- Error handling for invalid Base64

**Function Signature:**
```python
def decode_base64_audio(base64_string: str) -> bytes:
    """Decode Base64 to raw audio bytes"""
```

**Output:** Working Base64 decoder with error handling

---

## F2.2 - MP3 to WAV Converter
**Description:** Convert MP3 audio to WAV format at 16kHz mono for ML models.

**Input:** MP3 bytes
**Output:** WAV audio array at 16kHz

**Implementation Details:**
- Use pydub/librosa for conversion
- Resample to 16kHz (required by models)
- Convert to mono channel
- Normalize audio levels
- Handle corrupted audio files

**Function Signature:**
```python
def convert_to_wav_16khz(audio_bytes: bytes) -> np.ndarray:
    """Convert MP3 to 16kHz mono WAV array"""
```

**Output:** Reliable audio converter

---

## F2.3 - Audio Validation
**Description:** Validate audio meets requirements before processing.

**Validations:**
- [ ] Check audio duration (not too long/short)
- [ ] Verify audio has actual content (not silence)
- [ ] Validate audio format is MP3
- [ ] Check audio quality sufficient for detection

**Function Signature:**
```python
def validate_audio(audio_array: np.ndarray, sample_rate: int) -> ValidationResult:
    """Validate audio meets processing requirements"""
```

**Output:** Audio validator with proper error messages

---

# 📅 PHASE 3: ML Model Integration
**Duration:** Day 2-3 | **Priority:** Critical

## F3.1 - Language Detection Model
**Description:** Integrate SpeechBrain VoxLingua107 for language identification.

**Model:** `speechbrain/lang-id-voxlingua107-ecapa`

**Supported Languages Mapping:**
| Language | ISO Code | Model Output |
|----------|----------|--------------|
| Tamil | ta | Tamil |
| English | en | English |
| Hindi | hi | Hindi |
| Malayalam | ml | Malayalam |
| Telugu | te | Telugu |

**Implementation Details:**
- Load model on startup (singleton pattern)
- Cache model in memory for fast inference
- Map ISO codes to full language names
- Validate detected language is in supported list

**Function Signature:**
```python
class LanguageDetector:
    def detect(self, audio_array: np.ndarray) -> LanguageResult:
        """Detect language from audio"""
        # Returns: language_name, confidence, iso_code
```

**Output:** Working language detector with confidence scores

---

## F3.2 - Deepfake/AI Voice Detection Model
**Description:** Integrate MelodyMachine Deepfake Detection model.

**Model:** `MelodyMachine/Deepfake-audio-detection-V2`

**Model Specs:**
- Base: wav2vec2
- Accuracy: 99.73%
- Input: 16kHz audio
- Output: AI_GENERATED (1) or HUMAN (0)

**Implementation Details:**
- Load model on startup (singleton pattern)
- Use Wav2Vec2 feature extractor
- Get classification probabilities
- Calculate confidence score (0.0 - 1.0)

**Function Signature:**
```python
class DeepfakeDetector:
    def detect(self, audio_array: np.ndarray) -> DetectionResult:
        """Detect if audio is AI generated or Human"""
        # Returns: classification, confidence_score
```

**Output:** Working AI/Human voice detector

---

## F3.3 - Explanation Generator
**Description:** Generate human-readable explanations for detection results.

**Implementation Details:**
- Based on confidence score and classification
- Consider audio characteristics
- Provide meaningful reasons

**Example Explanations:**
| Classification | Confidence | Explanation |
|----------------|------------|-------------|
| AI_GENERATED | >0.9 | "Strong synthetic speech patterns detected with unnatural pitch consistency" |
| AI_GENERATED | 0.7-0.9 | "Moderate indicators of AI-generated speech, including regular prosody" |
| HUMAN | >0.9 | "Natural speech variations and authentic voice characteristics detected" |
| HUMAN | 0.7-0.9 | "Voice exhibits mostly natural patterns with minor irregularities" |

**Function Signature:**
```python
def generate_explanation(classification: str, confidence: float, 
                         audio_features: dict) -> str:
    """Generate explanation for the detection result"""
```

**Output:** Context-aware explanation generator

---

# 📅 PHASE 4: REST API Development
**Duration:** Day 3-4 | **Priority:** Critical

## F4.1 - API Key Authentication Middleware
**Description:** Implement API key validation middleware.

**Implementation Details:**
- Check `x-api-key` header
- Validate against stored API key
- Return 401 for invalid/missing key
- Secure comparison to prevent timing attacks

**Error Response:**
```json
{
    "status": "error",
    "message": "Invalid API key or malformed request"
}
```

**Function Signature:**
```python
async def verify_api_key(x_api_key: str = Header(...)) -> bool:
    """Validate API key from request header"""
```

**Output:** Secure API key authentication

---

## F4.2 - Request Schema Validation
**Description:** Define and validate request body using Pydantic.

**Request Schema:**
```python
class VoiceDetectionRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    audioFormat: Literal["mp3"]
    audioBase64: str
    
    @validator('audioBase64')
    def validate_base64(cls, v):
        # Validate Base64 format
```

**Output:** Type-safe request validation

---

## F4.3 - Response Schema
**Description:** Define response models for success and error cases.

**Success Response Schema:**
```python
class VoiceDetectionResponse(BaseModel):
    status: Literal["success"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float  # 0.0 to 1.0
    explanation: str
```

**Error Response Schema:**
```python
class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str
```

**Output:** Consistent response format

---

## F4.4 - Main Detection Endpoint
**Description:** Implement the core `/api/voice-detection` endpoint.

**Endpoint:** `POST /api/voice-detection`

**Flow:**
1. Validate API key
2. Parse and validate request body
3. Decode Base64 audio
4. Convert to 16kHz WAV
5. (Optional) Detect language & validate
6. Run AI/Human detection
7. Generate explanation
8. Return JSON response

**Implementation:**
```python
@router.post("/api/voice-detection")
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
) -> VoiceDetectionResponse:
    """Main voice detection endpoint"""
```

**Output:** Fully functional detection endpoint

---

## F4.5 - Error Handling
**Description:** Implement comprehensive error handling.

**Error Cases:**
| Error | HTTP Code | Message |
|-------|-----------|---------|
| Missing API Key | 401 | "API key required" |
| Invalid API Key | 401 | "Invalid API key" |
| Invalid Base64 | 400 | "Invalid Base64 audio data" |
| Unsupported Language | 400 | "Unsupported language" |
| Audio Too Long | 400 | "Audio exceeds maximum duration" |
| Processing Error | 500 | "Error processing audio" |

**Output:** Robust error handling with clear messages

---

# 📅 PHASE 5: Testing & Optimization
**Duration:** Day 4-5 | **Priority:** High

## F5.1 - Unit Tests
**Description:** Write unit tests for all components.

**Test Files:**
- `test_audio_processor.py` - Audio processing tests
- `test_language_detector.py` - Language detection tests
- `test_deepfake_detector.py` - AI detection tests
- `test_api.py` - API endpoint tests

**Test Cases:**
- [ ] Valid Base64 decoding
- [ ] Invalid Base64 handling
- [ ] Audio conversion accuracy
- [ ] Language detection for all 5 languages
- [ ] AI voice detection accuracy
- [ ] Human voice detection accuracy
- [ ] API key validation
- [ ] Request validation
- [ ] Response format verification

**Output:** >90% test coverage

---

## F5.2 - Integration Tests
**Description:** Test complete API flow end-to-end.

**Test Scenarios:**
- [ ] Full request with Tamil AI voice → Correct detection
- [ ] Full request with English Human voice → Correct detection
- [ ] Request without API key → 401 error
- [ ] Request with invalid Base64 → 400 error
- [ ] Concurrent requests handling

**Output:** Verified end-to-end functionality

---

## F5.3 - Performance Optimization
**Description:** Optimize for speed and memory efficiency.

**Optimizations:**
- [ ] Model loading on startup (not per request)
- [ ] GPU inference if available (CUDA)
- [ ] Audio processing in memory (no disk I/O)
- [ ] Response caching for identical requests
- [ ] Async processing where possible

**Target Performance:**
| Metric | Target |
|--------|--------|
| Response Time | < 3 seconds |
| Memory Usage | < 2GB |
| Concurrent Users | 10+ |

**Output:** Optimized API ready for production

---

# 📅 PHASE 6: Deployment & Documentation
**Duration:** Day 5-6 | **Priority:** High

## F6.1 - Docker Configuration
**Description:** Create Docker setup for easy deployment.

**Files:**
- `Dockerfile` - Container definition
- `docker-compose.yml` - Service orchestration

**Dockerfile Features:**
- Python 3.10 base image
- FFmpeg installation
- Model pre-download
- Optimized layer caching

**Output:** Production-ready Docker setup

---

## F6.2 - Cloud Deployment
**Description:** Deploy to cloud platform.

**Options:**
| Platform | Pros | Cons |
|----------|------|------|
| Hugging Face Spaces | Free, easy | Limited resources |
| Railway | Easy, scalable | Paid after limits |
| Render | Free tier, auto-deploy | Cold starts |
| AWS EC2 | Full control | Complex setup |

**Recommended:** Hugging Face Spaces (Free, ML-optimized)

**Output:** Live deployed API

---

## F6.3 - API Documentation
**Description:** Create comprehensive API documentation.

**Documentation Includes:**
- [ ] Endpoint description
- [ ] Request/Response examples
- [ ] Authentication guide
- [ ] Error codes reference
- [ ] Rate limits
- [ ] Code samples (cURL, Python, JavaScript)

**Output:** Complete API documentation

---

# ⏱️ Timeline Summary

| Phase | Duration | Features | Priority |
|-------|----------|----------|----------|
| Phase 1 | Day 1 | F1.1, F1.2, F1.3 | Critical |
| Phase 2 | Day 1-2 | F2.1, F2.2, F2.3 | Critical |
| Phase 3 | Day 2-3 | F3.1, F3.2, F3.3 | Critical |
| Phase 4 | Day 3-4 | F4.1, F4.2, F4.3, F4.4, F4.5 | Critical |
| Phase 5 | Day 4-5 | F5.1, F5.2, F5.3 | High |
| Phase 6 | Day 5-6 | F6.1, F6.2, F6.3 | High |

**Total Estimated Time:** 5-6 Days

---

# 📊 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| AI Detection Accuracy | >95% | Test dataset evaluation |
| Language Detection Accuracy | >90% | Test dataset evaluation |
| API Response Time | <3s | Load testing |
| API Uptime | >99% | Monitoring |
| Error Rate | <1% | Logging analysis |

---

# 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Build Docker image
docker build -t ai-voice-detection .

# Run Docker container
docker run -p 8000:8000 ai-voice-detection
```

---

**Ready to start implementation? Let's begin with Phase 1!** 🚀
