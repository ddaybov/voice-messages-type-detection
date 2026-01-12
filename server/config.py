"""
Configuration management for the voice messages type detection server.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    max_upload_mb: float = 20.0


@dataclass
class ModelConfig:
    """Model configuration."""
    models_dir: str = "./models"
    default_model: str = "logreg"
    ensemble_weights: str = "daybovnet:1,logreg:1"
    transformer_model: str = "cointegrated/rubert-tiny2"
    num_labels: int = 2


@dataclass
class ASRConfig:
    """ASR (Automatic Speech Recognition) configuration."""
    backend: str = "speech_recognition"  # speech_recognition, vosk, whisper
    language: str = "ru-RU"
    min_duration: float = 0.0
    min_words: int = 0
    whisper_model: str = "tiny"  # tiny, base, small, medium, large
    vosk_model_path: Optional[str] = None


@dataclass
class FFmpegConfig:
    """FFmpeg configuration."""
    ffmpeg_bin: Optional[str] = None
    ffprobe_bin: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    server: ServerConfig
    model: ModelConfig
    asr: ASRConfig
    ffmpeg: FFmpegConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            server=ServerConfig(
                host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")),
                log_level=os.getenv("LOG_LEVEL", "info"),
                max_upload_mb=float(os.getenv("MAX_UPLOAD_MB", "20")),
            ),
            model=ModelConfig(
                models_dir=os.getenv("MODELS_DIR", "./models"),
                default_model=os.getenv("DEFAULT_MODEL", "logreg"),
                ensemble_weights=os.getenv("ENSEMBLE_WEIGHTS", "daybovnet:1,logreg:1"),
                transformer_model=os.getenv("TRANSFORMER_MODEL", "cointegrated/rubert-tiny2"),
                num_labels=int(os.getenv("NUM_LABELS", "2")),
            ),
            asr=ASRConfig(
                backend=os.getenv("ASR_BACKEND", "speech_recognition"),
                language=os.getenv("ASR_LANGUAGE", "ru-RU"),
                min_duration=float(os.getenv("MIN_DURATION", "0") or "0"),
                min_words=int(os.getenv("MIN_WORDS", "0") or "0"),
                whisper_model=os.getenv("WHISPER_MODEL", "tiny"),
                vosk_model_path=os.getenv("VOSK_MODEL_PATH"),
            ),
            ffmpeg=FFmpegConfig(
                ffmpeg_bin=os.getenv("FFMPEG_BIN"),
                ffprobe_bin=os.getenv("FFPROBE_BIN"),
            ),
        )


# Global configuration instance
config = AppConfig.from_env()
