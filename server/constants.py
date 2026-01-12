"""
Constants for the voice messages type detection server.
"""

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = [
    'wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'wma'
]

# Supported languages for ASR
SUPPORTED_LANGUAGES = {
    "ru-RU": "Русский",
    "en-US": "English (US)",
}

# Model aliases (for backward compatibility)
MODEL_ALIASES = {
    "daybov": "daybovnet",
    "daybovnet": "daybovnet",
    "logreg": "logreg",
    "logistic": "logreg",
    "svm": "svm",
    "nb": "nb",
    "naive_bayes": "nb",
    "bert": "bert",
    "dimanet": "dimanet",
    "dima": "dimanet",
    "ensemble": "ensemble",
}

# Default values
DEFAULT_MAX_TEXT_PREVIEW_LENGTH = 100
DEFAULT_CONFIDENCE_BAR_LENGTH = 10
