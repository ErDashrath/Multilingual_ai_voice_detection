from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import librosa
import torch
import torch.nn.functional as F

# Load the fine-tuned XLSR model
model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

print("✅ Model Loaded Successfully")
# Add these imports if you haven't already

def predict_audio(file_path):
    # 1. Load the audio file (librosa automatically resamples to 16kHz)
    audio, sr = librosa.load(file_path, sr=16000)
    
    # 2. Tokenize (prepare the input for the model)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # 3. Predict (Forward pass)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # 4. Get Probabilities (Softmax)
    probabilities = F.softmax(logits, dim=-1)
    
    # 5. Get the label (Fake or Real)
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_id]
    confidence = probabilities[0][predicted_id].item()
    
    return predicted_label, confidence

# --- TEST AREA ---
# Replace 'test_audio.mp3' with a real file path on your Kali machine
try:
    label, score = predict_audio("audio/real_sample.m4a")
    print(f"Prediction: {label}")
    print(f"Confidence: {score:.4f}")
    
    # Check what label 0 and 1 actually mean for this specific model
    print("Label Map:", model.config.id2label) 
except Exception as e:
    print(f"Error testing audio: {e}")