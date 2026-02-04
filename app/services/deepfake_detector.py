"""Deepfake/AI Voice Detection Service using MelodyMachine model"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

from ..config import settings


@dataclass  
class DetectionResult:
    """Result from deepfake detection"""
    classification: str  # "AI_GENERATED" or "HUMAN"
    confidence: float    # 0.00 to 1.00
    raw_scores: dict     # Raw model scores


class DeepfakeDetector:
    """Detects AI-generated vs Human voice using wav2vec2"""
    
    _instance = None
    _model = None
    _feature_extractor = None
    _device = None
    _loaded = False
    
    # Local model path - models are pre-downloaded
    LOCAL_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "deepfake_detector"
    
    def __new__(cls):
        """Singleton pattern for model loading"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not DeepfakeDetector._loaded:
            self._load_model()
    
    def _load_model(self):
        """Load the deepfake detection model from local folder"""
        try:
            DeepfakeDetector._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model_path = str(self.LOCAL_MODEL_PATH)
            
            # Verify local model exists (safetensors or PyTorch bin)
            model_files = [
                "model.safetensors",
                "pytorch_model.bin",
                "tf_model.h5",
                "flax_model.msgpack",
            ]
            if not any((self.LOCAL_MODEL_PATH / name).exists() for name in model_files):
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Please download the model first."
                )
            
            print(f"📂 Loading deepfake model from: {model_path}")
            
            # Load feature extractor
            DeepfakeDetector._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            # Load model
            DeepfakeDetector._model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True
            )
            DeepfakeDetector._model.to(DeepfakeDetector._device)
            DeepfakeDetector._model.eval()
            DeepfakeDetector._loaded = True
            
            print(f"✓ Deepfake model loaded | Device: {DeepfakeDetector._device}")
            
        except Exception as e:
            print(f"✗ Failed to load deepfake model: {str(e)}")
            raise RuntimeError(f"Failed to load deepfake detection model: {str(e)}")
    
    def detect(self, audio_array: np.ndarray) -> DetectionResult:
        """
        Detect if audio is AI-generated or Human
        
        Args:
            audio_array: Numpy array of audio samples at 16kHz
            
        Returns:
            DetectionResult with classification and confidence (0.00-1.00)
        """
        try:
            # Prepare input features
            inputs = DeepfakeDetector._feature_extractor(
                audio_array,
                sampling_rate=settings.SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(DeepfakeDetector._device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = DeepfakeDetector._model(**inputs)
                logits = outputs.logits
            
            # Get probabilities using softmax
            probabilities = torch.softmax(logits, dim=-1)
            
            # Model labels: {0: 'fake', 1: 'real'}
            # Index 0 = fake/AI_GENERATED
            # Index 1 = real/HUMAN
            fake_prob = float(probabilities[0][0].item())
            real_prob = float(probabilities[0][1].item())
            
            # Classification based on higher probability
            if fake_prob > real_prob:
                classification = "AI_GENERATED"
                confidence = round(fake_prob, 2)  # 0.00 to 1.00
            else:
                classification = "HUMAN"
                confidence = round(real_prob, 2)  # 0.00 to 1.00
            
            raw_scores = {
                "ai_score": round(fake_prob, 4),
                "human_score": round(real_prob, 4)
            }
            
            return DetectionResult(
                classification=classification,
                confidence=confidence,
                raw_scores=raw_scores
            )
            
        except Exception as e:
            raise RuntimeError(f"Deepfake detection failed: {str(e)}")
