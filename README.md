# AI Voice Detection API

🎙️ Detects whether a voice recording is **AI-generated** or **Human** across 5 Indian languages.

## 🌍 Supported Languages

- Tamil
- English
- Hindi
- Malayalam
- Telugu

## 🤖 Models Used

| Model | Purpose | Accuracy |
|-------|---------|----------|
| `MelodyMachine/Deepfake-audio-detection-V2` | AI/Human Detection | **99.73%** |
| `speechbrain/lang-id-voxlingua107-ecapa` | Language Detection | **93.3%** |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

The models are not included in the repo due to size. Download them first:

```bash
# Create models directory
mkdir -p models/deepfake_detector

# Download deepfake detection model from HuggingFace
python -c "
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

model_name = 'MelodyMachine/Deepfake-audio-detection-V2'
save_path = './models/deepfake_detector'

print('Downloading model...')
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

model.save_pretrained(save_path)
feature_extractor.save_pretrained(save_path)
print(f'Model saved to {save_path}')
"
```

### 3. Run the Server

```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the run script
chmod +x run.sh
./run.sh
```

### 4. Access the API

- **API Endpoint:** http://localhost:8000/api/voice-detection
- **Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## 📡 API Usage

### Request

```bash
curl -X POST https://your-domain.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'
```

### Response (Success)

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Strong synthetic speech indicators detected: unnatural pitch consistency, flat energy profile. Voice clearly exhibits AI-generated characteristics."
}
```

### Response (Error)

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## 🔐 Authentication

All requests require an API key via the `x-api-key` header.

Default API Key (for testing): `sk_test_123456789`

## 📁 Project Structure

```
aivoice_detection/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration settings
│   ├── routes/
│   │   └── voice_detection.py
│   ├── services/
│   │   ├── audio_processor.py
│   │   ├── language_detector.py
│   │   ├── deepfake_detector.py
│   │   └── explanation_generator.py
│   ├── models/
│   │   └── schemas.py
│   └── middleware/
│       └── auth.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── .env
└── README.md
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t ai-voice-detection .

# Run container
docker run -p 8000:8000 ai-voice-detection
```

## 📊 Explanation Features

The API provides detailed explanations based on audio analysis:

### AI-Generated Indicators
- Unnatural pitch consistency
- Flat energy profile
- Missing natural speech pauses
- Overly regular speech rhythm
- Synthetic speech artifacts

### Human Voice Indicators
- Natural pitch variations
- Dynamic energy patterns
- Natural breathing patterns
- Expressive pitch range
- Organic timing variations

## 📝 License

MIT License
