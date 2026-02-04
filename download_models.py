"""
Download and save models locally for faster loading
"""

import os
from pathlib import Path

# Set the models directory
MODELS_DIR = Path(__file__).parent / "models"
DEEPFAKE_DIR = MODELS_DIR / "deepfake_detector"
LANGUAGE_DIR = MODELS_DIR / "language_detector"

def download_deepfake_model():
    """Download MelodyMachine/Deepfake-audio-detection-V2"""
    print("=" * 60)
    print("📥 Downloading Deepfake Detection Model...")
    print("   Model: MelodyMachine/Deepfake-audio-detection-V2")
    print("=" * 60)
    
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    
    model_name = "MelodyMachine/Deepfake-audio-detection-V2"
    
    # Download and save feature extractor
    print("   Downloading feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    feature_extractor.save_pretrained(DEEPFAKE_DIR)
    
    # Download and save model
    print("   Downloading model weights...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(DEEPFAKE_DIR)
    
    print(f"✅ Saved to: {DEEPFAKE_DIR}")
    print(f"   Model labels: {model.config.id2label}")
    

def download_language_model():
    """Download speechbrain/lang-id-voxlingua107-ecapa"""
    print("=" * 60)
    print("📥 Downloading Language Detection Model...")
    print("   Model: speechbrain/lang-id-voxlingua107-ecapa")
    print("=" * 60)
    
    from speechbrain.inference.classifiers import EncoderClassifier
    
    # Download and save
    print("   Downloading model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=str(LANGUAGE_DIR)
    )
    
    print(f"✅ Saved to: {LANGUAGE_DIR}")


def main():
    print("\n" + "=" * 60)
    print("🚀 AI Voice Detection - Model Downloader")
    print("=" * 60 + "\n")
    
    # Create directories
    DEEPFAKE_DIR.mkdir(parents=True, exist_ok=True)
    LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download models
    download_deepfake_model()
    print()
    download_language_model()
    
    print("\n" + "=" * 60)
    print("✅ All models downloaded successfully!")
    print("=" * 60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print("\nYou can now run the API with local models.")


if __name__ == "__main__":
    main()
