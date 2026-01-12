
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

    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return SUPPORTED_AUDIO_FORMATS.copy()

    def get_supported_languages(self) -> dict:
        """Get dictionary of supported languages."""
        return SUPPORTED_LANGUAGES.copy()

    def to_wav(self, file_path: str) -> Optional[str]:
        try:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            fd, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            logger.exception("FFmpeg convert error: %s", e)
            return None

    def transcribe(self, wav_path: str, lang: str) -> Optional[ASRResult]:
        lang = lang if lang in SUPPORTED_LANGUAGES else "ru-RU"
        if self.asr_backend == "speech_recognition":
            return self._transcribe_speech_recognition(wav_path, lang)
        if self.asr_backend == "vosk":
            return self._transcribe_vosk(wav_path, lang)
        if self.asr_backend == "whisper":
            return self._transcribe_whisper(wav_path, lang)
        logger.error("Unknown ASR_BACKEND=%s", self.asr_backend); return None

    def _transcribe_speech_recognition(self, wav_path: str, lang: str) -> Optional[ASRResult]:
        r = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
        except Exception as e:
            logger.error(f"Error reading audio file {wav_path}: {e}")
            return None
        
        text = ""
        backend_name = "SpeechRecognition(Google)"
        
        # Попытка 1: Google Speech Recognition
        try:
            logger.info(f"Attempting Google Speech Recognition for language {lang}")
            text = r.recognize_google(audio, language=lang)
            logger.info(f"Google Speech Recognition succeeded: {len(text)} characters")
        except sr.RequestError as e:
            logger.warning(f"Google Speech Recognition request failed: {e}, trying PocketSphinx...")
            # Попытка 2: PocketSphinx (офлайн)
            try:
                text = r.recognize_sphinx(audio, language=lang.split('-')[0])
                backend_name = "SpeechRecognition(Sphinx)"
                logger.info(f"PocketSphinx succeeded: {len(text)} characters")
            except Exception as e:
                logger.exception(f"PocketSphinx failed: {e}")
                return None
        except sr.UnknownValueError as e:
            logger.warning(f"Google Speech Recognition could not understand audio: {e}, trying PocketSphinx...")
            # Попытка fallback на PocketSphinx при UnknownValueError
            try:
                text = r.recognize_sphinx(audio, language=lang.split('-')[0])
                backend_name = "SpeechRecognition(Sphinx)"
                logger.info(f"PocketSphinx fallback succeeded: {len(text)} characters")
            except Exception as e2:
                logger.warning(f"PocketSphinx fallback also failed: {e2}")
                text = ""
        except Exception as e:
            logger.exception(f"Unexpected error in speech recognition: {e}")
            return None
        
        try:
            duration = AudioSegment.from_wav(wav_path).duration_seconds
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            duration = 0.0
        
        words = len(text.strip().split()) if text else 0
        logger.info(f"ASR result: text_length={len(text)}, words={words}, duration={duration:.2f}s, backend={backend_name}")
        
        return ASRResult(text=text, duration=float(duration), word_count=words, backend=backend_name)

    def _transcribe_vosk(self, wav_path: str, lang: str) -> Optional[ASRResult]:
        try:
            from vosk import Model, KaldiRecognizer
            import json as _json, wave
        except Exception:
            logger.exception("Please install vosk to use ASR_BACKEND=vosk"); return None
        model_path = os.getenv("VOSK_MODEL_PATH")
        if not model_path or not os.path.exists(model_path):
            logger.error("VOSK_MODEL_PATH not set or missing: %s", model_path); return None
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(Model(model_path), wf.getframerate()); rec.SetWords(True)
        text = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0: break
            if rec.AcceptWaveform(data):
                res = rec.Result(); text.append(_json.loads(res).get("text",""))
        final = _json.loads(rec.FinalResult()).get("text","")
        full_text = " ".join([*text, final]).strip()
        duration = AudioSegment.from_wav(wav_path).duration_seconds
        return ASRResult(text=full_text, duration=float(duration), word_count=len(full_text.split()), backend="Vosk")

    def _transcribe_whisper(self, wav_path: str, lang: str) -> Optional[ASRResult]:
        try:
            import whisper
        except Exception as e:
            logger.exception(f"Please install openai-whisper to use ASR_BACKEND=whisper: {e}")
            return None
        
        model_name = self.config.whisper_model
        lang_code = lang.split('-')[0] if '-' in lang else lang
        
        try:
            logger.info(f"Loading Whisper model: {model_name}")
            model = whisper.load_model(model_name)
            logger.info(f"Whisper model {model_name} loaded successfully")
        except Exception as e:
            logger.exception(f"Error loading Whisper model {model_name}: {e}")
            return None
        
        try:
            logger.info(f"Transcribing with Whisper: {wav_path}, lang={lang_code}, model={model_name}")
            result = model.transcribe(wav_path, language=lang_code if lang_code in whisper.tokenizer.LANGUAGES else None)
            text = result.get("text", "").strip()
            logger.info(f"Whisper transcription completed: text_length={len(text)}")
        except Exception as e:
            logger.exception(f"Error transcribing with Whisper: {e}")
            return None
        
        try:
            duration = AudioSegment.from_wav(wav_path).duration_seconds
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            duration = result.get("duration", 0.0) if isinstance(result, dict) and "duration" in result else 0.0
        
        words = len(text.split()) if text else 0
        logger.info(f"ASR result: text_length={len(text)}, words={words}, duration={duration:.2f}s, backend=Whisper({model_name})")
        
        return ASRResult(text=text, duration=float(duration), word_count=words, backend=f"Whisper({model_name})")

audio_service = AudioService()
