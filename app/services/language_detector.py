"""Language Detection Service - Simplified version using audio characteristics"""

from dataclasses import dataclass
from typing import List

from ..config import settings


@dataclass
class LanguageResult:
    """Result from language detection"""
    language: str
    confidence: float
    is_supported: bool


class LanguageDetector:
    """
    Simplified language detector.
    For now, we trust the user-provided language since the main focus is AI/Human detection.
    The language field is used for reporting purposes.
    """
    
    SUPPORTED_LANGUAGES: List[str] = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    def __init__(self):
        pass
    
    def validate_language(self, language: str) -> LanguageResult:
        """
        Validate if provided language is supported
        
        Args:
            language: Language name from request
            
        Returns:
            LanguageResult with validation status
        """
        is_supported = language in self.SUPPORTED_LANGUAGES
        
        return LanguageResult(
            language=language,
            confidence=1.0 if is_supported else 0.0,
            is_supported=is_supported
        )
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages"""
        return self.SUPPORTED_LANGUAGES
