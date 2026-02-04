
import os
import tempfile
import logging
from dataclasses import dataclass
from typing import Optional

from pydub import AudioSegment
import speech_recognition as sr

from .config import config
from .constants import SUPPORTED_AUDIO_FORMATS, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

# Configure FFmpeg paths
if config.ffmpeg.ffmpeg_bin and os.path.exists(config.ffmpeg.ffmpeg_bin):
    AudioSegment.converter = config.ffmpeg.ffmpeg_bin
if config.ffmpeg.ffprobe_bin and os.path.exists(config.ffmpeg.ffprobe_bin):
    AudioSegment.ffprobe = config.ffmpeg.ffprobe_bin

@dataclass
class ASRResult:
    text: str
    duration: float
    word_count: int
    backend: str

class AudioService:
    """Service for audio processing and speech recognition."""

    def __init__(self, asr_config=None):
        """
        Initialize AudioService.

        Args:
            asr_config: ASRConfig instance. If None, uses global config.
        """
        self.config = asr_config or config.asr
        self.min_duration = self.config.min_duration
        self.min_words = self.config.min_words
        self.asr_backend = self.config.backend
        self._whisper_model = None
        self._whisper_model_name = None

    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return SUPPORTED_AUDIO_FORMATS.copy()

    def get_supported_languages(self) -> dict:
        """Get dictionary of supported languages."""
        return SUPPORTED_LANGUAGES.copy()

    def to_wav(self, file_path: str) -> Optional[str]:
        try:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            fd, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            logger.exception("FFmpeg convert error: %s", e)
            return None

    def transcribe(self, wav_path: str, lang: str, src_path: Optional[str] = None) -> Optional[ASRResult]:
        lang = lang if lang in SUPPORTED_LANGUAGES else "ru-RU"
        if self.asr_backend == "speech_recognition":
            return self._transcribe_speech_recognition(wav_path, lang)
        if self.asr_backend == "vosk":
            return self._transcribe_vosk(wav_path, lang)
        if self.asr_backend == "whisper":
            audio_path = src_path if src_path else wav_path
            if src_path:
                logger.info("Whisper: using original file %s (bypass pydub conversion)", src_path)
            result = self._transcribe_whisper(audio_path, lang)
            if result and (not result.text or not result.text.strip()):
                logger.info("Whisper returned empty text, trying Google Speech Recognition fallback")
                fallback = self._transcribe_speech_recognition(wav_path, lang)
                if fallback and fallback.text and fallback.text.strip():
                    return ASRResult(
                        text=fallback.text,
                        duration=fallback.duration,
                        word_count=fallback.word_count,
                        backend="%s+Google" % result.backend,
                    )
            return result
        logger.error("Unknown ASR_BACKEND=%s", self.asr_backend)
        return None

    def _transcribe_speech_recognition(self, wav_path: str, lang: str) -> Optional[ASRResult]:
        r = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
        except Exception as e:
            logger.error("Error reading audio file %s: %s", wav_path, e)
            return None

        text = ""
        backend_name = "SpeechRecognition(Google)"

        try:
            logger.info("Attempting Google Speech Recognition for language %s", lang)
            text = r.recognize_google(audio, language=lang)
            logger.info("Google Speech Recognition succeeded: %s characters", len(text))
        except sr.RequestError as e:
            logger.warning("Google Speech Recognition request failed: %s, trying PocketSphinx...", e)
            try:
                text = r.recognize_sphinx(audio, language=lang.split('-')[0])
                backend_name = "SpeechRecognition(Sphinx)"
                logger.info("PocketSphinx succeeded: %s characters", len(text))
            except Exception as e2:
                logger.exception("PocketSphinx failed: %s", e2)
                return None
        except sr.UnknownValueError as e:
            logger.warning("Google Speech Recognition could not understand audio: %s, trying PocketSphinx...", e)
            try:
                text = r.recognize_sphinx(audio, language=lang.split('-')[0])
                backend_name = "SpeechRecognition(Sphinx)"
                logger.info("PocketSphinx fallback succeeded: %s characters", len(text))
            except Exception as e2:
                logger.warning("PocketSphinx fallback also failed: %s", e2)
                text = ""
        except Exception as e:
            logger.exception("Unexpected error in speech recognition: %s", e)
            return None

        try:
            duration = AudioSegment.from_wav(wav_path).duration_seconds
        except Exception as e:
            logger.error("Error getting audio duration: %s", e)
            duration = 0.0

        words = len(text.strip().split()) if text else 0
        logger.info("ASR result: text_length=%s, words=%s, duration=%.2fs, backend=%s", len(text), words, duration, backend_name)

        return ASRResult(text=text, duration=float(duration), word_count=words, backend=backend_name)

    def _transcribe_vosk(self, wav_path: str, lang: str) -> Optional[ASRResult]:
        try:
            from vosk import Model, KaldiRecognizer
            import json as _json
            import wave
        except Exception:
            logger.exception("Please install vosk to use ASR_BACKEND=vosk")
            return None
        model_path = os.getenv("VOSK_MODEL_PATH")
        if not model_path or not os.path.exists(model_path):
            logger.error("VOSK_MODEL_PATH not set or missing: %s", model_path)
            return None
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(Model(model_path), wf.getframerate())
        rec.SetWords(True)
        text = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = rec.Result()
                text.append(_json.loads(res).get("text", ""))
        final = _json.loads(rec.FinalResult()).get("text", "")
        full_text = " ".join([*text, final]).strip()
        duration = AudioSegment.from_wav(wav_path).duration_seconds
        return ASRResult(text=full_text, duration=float(duration), word_count=len(full_text.split()), backend="Vosk")

    def _transcribe_whisper(self, audio_path: str, lang: str) -> Optional[ASRResult]:
        try:
            import whisper
        except Exception as e:
            logger.exception("Please install openai-whisper to use ASR_BACKEND=whisper: %s", e)
            return None

        model_name = self.config.whisper_model
        lang_code = lang.split('-')[0] if '-' in lang else lang

        try:
            if self._whisper_model is None or self._whisper_model_name != model_name:
                logger.info("Loading Whisper model: %s", model_name)
                self._whisper_model = whisper.load_model(model_name)
                self._whisper_model_name = model_name
                logger.info("Whisper model %s loaded successfully", model_name)
            model = self._whisper_model
        except Exception as e:
            logger.exception("Error loading Whisper model %s: %s", model_name, e)
            return None

        try:
            logger.info("Transcribing with Whisper: %s, lang=%s, model=%s", audio_path, lang_code, model_name)
            result = model.transcribe(
                audio_path,
                language=lang_code if lang_code in whisper.tokenizer.LANGUAGES else None,
                fp16=False,
                temperature=0.0,
                no_speech_threshold=0.0,
                initial_prompt="Привет, один два три." if lang_code == "ru" else None,
            )
            text = result.get("text", "").strip()
            logger.info("Whisper transcription completed: text_length=%s", len(text))
            if not text and lang_code == "ru":
                logger.info("Whisper returned empty with language=ru, retrying with language=None")
                result = model.transcribe(
                    audio_path,
                    language=None,
                    fp16=False,
                    temperature=0.0,
                    no_speech_threshold=0.0,
                )
                text = result.get("text", "").strip()
                logger.info("Whisper (auto lang) completed: text_length=%s", len(text))
        except Exception as e:
            logger.exception("Error transcribing with Whisper: %s", e)
            return None

        try:
            duration = AudioSegment.from_file(audio_path).duration_seconds
        except Exception as e:
            logger.error("Error getting audio duration: %s", e)
            duration = result.get("duration", 0.0) if isinstance(result, dict) and "duration" in result else 0.0

        words = len(text.split()) if text else 0
        logger.info("ASR result: text_length=%s, words=%s, duration=%.2fs, backend=Whisper(%s)", len(text), words, duration, model_name)

        return ASRResult(text=text, duration=float(duration), word_count=words, backend="Whisper(%s)" % model_name)


audio_service = AudioService()
