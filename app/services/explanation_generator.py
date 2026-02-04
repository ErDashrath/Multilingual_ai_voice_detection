"""Explanation Generator - Provides meaningful reasons for detection results"""

from dataclasses import dataclass
from typing import Optional
from .audio_processor import AudioFeatures


class ExplanationGenerator:
    """
    Generates human-readable explanations for AI/Human voice detection.
    
    Analyzes audio features to provide context-aware reasoning:
    - Pitch patterns (monotonic = AI indicator)
    - Energy variations (flat = AI indicator) 
    - Natural pauses (present = Human indicator)
    - Speech rhythm (regular = AI, varied = Human)
    """
    
    # AI-Generated voice indicators
    AI_INDICATORS = {
        "monotonic_pitch": "Unnatural pitch consistency detected with minimal variation",
        "flat_energy": "Flat energy profile without natural dynamics",
        "no_pauses": "Missing natural speech pauses and breathing patterns",
        "regular_rhythm": "Overly regular speech rhythm and timing patterns",
        "synthetic_artifacts": "Synthetic speech artifacts and unnatural transitions detected",
        "perfect_prosody": "Unnaturally perfect prosody without micro-variations",
        "robotic_pattern": "Robotic speech patterns with mechanical consistency",
        "uniform_formants": "Uniform formant structure lacking natural variability"
    }
    
    # Human voice indicators
    HUMAN_INDICATORS = {
        "varied_pitch": "Natural pitch variations and intonation patterns detected",
        "dynamic_energy": "Dynamic energy profile with natural emphasis and stress",
        "natural_pauses": "Natural speech pauses and breathing patterns present",
        "irregular_rhythm": "Natural speech rhythm with organic timing variations",
        "micro_variations": "Subtle micro-variations characteristic of human speech",
        "authentic_prosody": "Authentic prosodic features with natural expressiveness",
        "voice_texture": "Natural voice texture with expected acoustic properties",
        "emotional_cues": "Emotional cues and natural voice modulation detected"
    }
    
    def generate(
        self, 
        classification: str, 
        confidence: float,
        audio_features: Optional[AudioFeatures] = None
    ) -> str:
        """
        Generate explanation for the detection result
        
        Args:
            classification: "AI_GENERATED" or "HUMAN"
            confidence: Confidence score (0.0 to 1.0)
            audio_features: Extracted audio features for detailed analysis
            
        Returns:
            Human-readable explanation string
        """
        if classification == "AI_GENERATED":
            return self._generate_ai_explanation(confidence, audio_features)
        else:
            return self._generate_human_explanation(confidence, audio_features)
    
    def _generate_ai_explanation(
        self, 
        confidence: float, 
        features: Optional[AudioFeatures]
    ) -> str:
        """Generate explanation for AI-generated voice detection"""
        
        reasons = []
        
        if features:
            # Check pitch consistency (AI tends to have monotonic pitch)
            if features.is_monotonic or features.pitch_std < 20:
                reasons.append("unnatural pitch consistency")
            
            # Check energy variation (AI tends to have flat energy)
            if features.energy_std < 0.02:
                reasons.append("flat energy profile")
            
            # Check for natural pauses (AI often lacks them)
            if not features.has_natural_pauses:
                reasons.append("missing natural speech pauses")
            
            # Check pitch range (AI often has limited range)
            if features.pitch_range < 50:
                reasons.append("limited pitch range")
        
        # Build explanation based on confidence and detected patterns
        if confidence >= 0.9:
            strength = "Strong"
            certainty = "clearly"
        elif confidence >= 0.75:
            strength = "Moderate"
            certainty = "likely"
        else:
            strength = "Weak"
            certainty = "possibly"
        
        if reasons:
            # Use detected audio features
            reason_str = ", ".join(reasons[:2])  # Use top 2 reasons
            explanation = f"{strength} synthetic speech indicators detected: {reason_str}. Voice {certainty} exhibits AI-generated characteristics."
        else:
            # Default explanations based on confidence
            if confidence >= 0.9:
                explanation = "Unnatural pitch consistency and robotic speech patterns detected. Strong indicators of AI-generated synthetic speech."
            elif confidence >= 0.75:
                explanation = "Moderate indicators of AI-generated speech including regular prosody and uniform acoustic patterns."
            else:
                explanation = "Some synthetic speech characteristics detected. Voice shows patterns consistent with AI generation."
        
        return explanation
    
    def _generate_human_explanation(
        self, 
        confidence: float, 
        features: Optional[AudioFeatures]
    ) -> str:
        """Generate explanation for human voice detection"""
        
        reasons = []
        
        if features:
            # Check for natural pitch variation
            if not features.is_monotonic and features.pitch_std > 20:
                reasons.append("natural pitch variations")
            
            # Check for dynamic energy
            if features.energy_std > 0.02:
                reasons.append("dynamic energy patterns")
            
            # Check for natural pauses
            if features.has_natural_pauses:
                reasons.append("natural breathing patterns")
            
            # Check pitch range
            if features.pitch_range > 50:
                reasons.append("expressive pitch range")
        
        # Build explanation based on confidence and detected patterns
        if confidence >= 0.9:
            strength = "Strong"
            certainty = "clearly"
        elif confidence >= 0.75:
            strength = "Clear"
            certainty = "likely"
        else:
            strength = "Moderate"
            certainty = "appears to be"
        
        if reasons:
            reason_str = ", ".join(reasons[:2])
            explanation = f"{strength} human voice characteristics: {reason_str}. Voice {certainty} authentic human speech."
        else:
            if confidence >= 0.9:
                explanation = "Natural speech variations and authentic voice characteristics detected. Voice exhibits genuine human speech patterns."
            elif confidence >= 0.75:
                explanation = "Voice shows natural variations in pitch, rhythm, and energy consistent with authentic human speech."
            else:
                explanation = "Voice exhibits characteristics typical of human speech with natural acoustic properties."
        
        return explanation
    
    def get_detailed_analysis(
        self, 
        features: AudioFeatures,
        classification: str
    ) -> dict:
        """
        Get detailed feature analysis (for debugging/logging)
        
        Args:
            features: Extracted audio features
            classification: Detection result
            
        Returns:
            Dictionary with detailed analysis
        """
        return {
            "classification": classification,
            "feature_analysis": {
                "pitch": {
                    "mean": features.pitch_mean,
                    "std": features.pitch_std,
                    "range": features.pitch_range,
                    "is_monotonic": features.is_monotonic,
                    "indicator": "AI" if features.is_monotonic else "Human"
                },
                "energy": {
                    "mean": features.energy_mean,
                    "std": features.energy_std,
                    "indicator": "AI" if features.energy_std < 0.02 else "Human"
                },
                "rhythm": {
                    "tempo": features.tempo,
                    "zero_crossing_rate": features.zero_crossing_rate
                },
                "pauses": {
                    "has_natural_pauses": features.has_natural_pauses,
                    "indicator": "Human" if features.has_natural_pauses else "AI"
                },
                "spectral": {
                    "centroid": features.spectral_centroid,
                    "rolloff": features.spectral_rolloff
                }
            },
            "duration_seconds": features.duration
        }
