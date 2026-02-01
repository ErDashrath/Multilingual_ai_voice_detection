from gtts import gTTS
import os

# Ensure audio directory exists
os.makedirs("audio", exist_ok=True)

# --- Generate FAKE Sample ---
text = "This is a computer generated voice used to test the deepfake detection system."
tts = gTTS(text=text, lang='en')
tts.save("audio/fake_sample.mp3")
print("🤖 Generated 'audio/fake_sample.mp3'")