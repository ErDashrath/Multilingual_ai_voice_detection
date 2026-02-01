from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import librosa
import torch

# Load the fine-tuned XLSR model
model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

print("✅ Model Loaded Successfully")