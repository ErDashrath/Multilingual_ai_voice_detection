"""Audio processing service - Base64 decoding, format conversion, validation"""

import base64
import io
import numpy as np
import librosa
from pydub import AudioSegment
from typing import Tuple, Dict
from dataclasses import dataclass

from ..config import settings


@dataclass
class AudioFeatures:
    """Audio features extracted for analysis"""
    duration: float
    pitch_mean: float
    pitch_std: float
    pitch_range: float
    energy_mean: float
    energy_std: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_rolloff: float
    tempo: float
    is_monotonic: bool
    has_natural_pauses: bool


class AudioProcessor:
    """Handles audio processing operations"""
    
    def __init__(self):
        self.sample_rate = settings.SAMPLE_RATE
        self.max_duration = settings.MAX_AUDIO_DURATION_SECONDS
    
    def decode_base64(self, base64_string: str) -> bytes:
        """
        Decode Base64 encoded audio to raw bytes
        
        Args:
            base64_string: Base64 encoded audio string
            
        Returns:
            Raw audio bytes
            
        Raises:
            ValueError: If Base64 decoding fails
        """
        try:
            # Remove any whitespace or newlines
            base64_string = base64_string.strip().replace("\n", "").replace("\r", "")
            
            # Add padding if necessary
            padding = 4 - len(base64_string) % 4
            if padding != 4:
                base64_string += "=" * padding
            
            return base64.b64decode(base64_string)
        except Exception as e:
            raise ValueError(f"Failed to decode Base64 audio: {str(e)}")
    
    def convert_to_wav_array(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert MP3 bytes to WAV numpy array at 16kHz mono
        
        Args:
            audio_bytes: Raw MP3 audio bytes
            
        Returns:
            Numpy array of audio samples at 16kHz
            
        Raises:
            ValueError: If audio conversion fails
        """
        try:
            # Load MP3 using pydub
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes), 
                format="mp3"
            )
            
            # Convert to mono
            audio_segment = audio_segment.set_channels(1)
            
            # Convert to 16kHz
            audio_segment = audio_segment.set_frame_rate(self.sample_rate)
            
            # Get raw samples as numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Normalize to float32 [-1, 1]
            samples = samples.astype(np.float32) / 32768.0
            
            return samples
            
        except Exception as e:
            raise ValueError(f"Failed to convert audio: {str(e)}")
    
    def validate_audio(self, audio_array: np.ndarray) -> Dict[str, any]:
        """
        Validate audio meets requirements
        
        Args:
            audio_array: Numpy array of audio samples
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValueError: If audio validation fails
        """
        # Calculate duration
        duration = len(audio_array) / self.sample_rate
        
        # Check minimum duration (at least 0.5 seconds)
        if duration < 0.5:
            raise ValueError("Audio too short. Minimum duration is 0.5 seconds")
        
        # Check maximum duration
        if duration > self.max_duration:
            raise ValueError(f"Audio too long. Maximum duration is {self.max_duration} seconds")
        
        # Check if audio has content (not silence)
        rms = np.sqrt(np.mean(audio_array ** 2))
        if rms < 0.001:
            raise ValueError("Audio appears to be silent or nearly silent")
        
        return {
            "valid": True,
            "duration": duration,
            "rms_level": float(rms)
        }
    
    def extract_features(self, audio_array: np.ndarray) -> AudioFeatures:
        """
        Extract audio features for explanation generation
        
        Args:
            audio_array: Numpy array of audio samples
            
        Returns:
            AudioFeatures dataclass with extracted features
        """
        try:
            # Duration
            duration = len(audio_array) / self.sample_rate
            
            # Pitch analysis using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio_array, 
                sr=self.sample_rate,
                fmin=50,
                fmax=500
            )
            
            # Get pitch values where magnitude is significant
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            pitch_values = np.array(pitch_values) if pitch_values else np.array([0])
            
            pitch_mean = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))
            pitch_range = float(np.max(pitch_values) - np.min(pitch_values)) if len(pitch_values) > 1 else 0
            
            # Energy analysis
            energy = librosa.feature.rms(y=audio_array)[0]
            energy_mean = float(np.mean(energy))
            energy_std = float(np.std(energy))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            zero_crossing_rate = float(np.mean(zcr))
            
            # Spectral features
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=self.sample_rate)))
            spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio_array, sr=self.sample_rate)))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_array, sr=self.sample_rate)
            tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0]) if len(tempo) > 0 else 0.0
            
            # Check for monotonic pitch (AI indicator)
            is_monotonic = pitch_std < 15 if pitch_mean > 0 else False
            
            # Check for natural pauses (Human indicator)
            silence_threshold = 0.01
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)  # 10ms hop
            
            frames = librosa.util.frame(audio_array, frame_length=frame_length, hop_length=hop_length)
            frame_energy = np.mean(frames ** 2, axis=0)
            silence_frames = np.sum(frame_energy < silence_threshold)
            total_frames = len(frame_energy)
            
            has_natural_pauses = (silence_frames / total_frames) > 0.05 and (silence_frames / total_frames) < 0.4
            
            return AudioFeatures(
                duration=duration,
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                pitch_range=pitch_range,
                energy_mean=energy_mean,
                energy_std=energy_std,
                zero_crossing_rate=zero_crossing_rate,
                spectral_centroid=spectral_centroid,
                spectral_rolloff=spectral_rolloff,
                tempo=tempo,
                is_monotonic=is_monotonic,
                has_natural_pauses=has_natural_pauses
            )
            
        except Exception as e:
            # Return default features if extraction fails
            return AudioFeatures(
                duration=len(audio_array) / self.sample_rate,
                pitch_mean=0,
                pitch_std=0,
                pitch_range=0,
                energy_mean=0,
                energy_std=0,
                zero_crossing_rate=0,
                spectral_centroid=0,
                spectral_rolloff=0,
                tempo=0,
                is_monotonic=False,
                has_natural_pauses=True
            )
    
    def process(self, base64_audio: str) -> Tuple[np.ndarray, AudioFeatures]:
        """
        Complete audio processing pipeline
        
        Args:
            base64_audio: Base64 encoded MP3 audio
            
        Returns:
            Tuple of (audio_array, audio_features)
        """
        # Decode Base64
        audio_bytes = self.decode_base64(base64_audio)
        
        # Convert to WAV array
        audio_array = self.convert_to_wav_array(audio_bytes)
        
        # Validate
        self.validate_audio(audio_array)
        
        # Extract features for explanation
        features = self.extract_features(audio_array)
        
        return audio_array, features
