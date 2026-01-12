"""
FastAPI server for voice messages type detection.
"""

import os
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import config
from .audio_processor import AudioService
from .classifier import TextClassifier
from .utils import cleanup_files

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.server.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize services
audio_service = AudioService()
text_classifier = TextClassifier()

class Health(BaseModel):
    status: str = "ok"

class PredictResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    label: Optional[int] = None
    label_name: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    word_count: Optional[int] = None
    model: Optional[str] = None
    asr_backend: Optional[str] = None
    error: Optional[str] = None

app = FastAPI(
    title="Speech Style Classifier",
    version="2.0.0",
    description="API for classifying voice messages as formal or informal style"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=Health)
async def health():
    return Health(status="ok")

@app.get("/supported_formats")
async def supported_formats():
    return {
        "audio_formats": audio_service.get_supported_formats(),
        "languages": audio_service.get_supported_languages(),
        "status": "ok",
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    lang: str = Form("ru-RU"),
    model: str | None = Form(None),
) -> PredictResponse:
    """
    Predict voice message style (formal/informal).
    
    Args:
        file: Audio file to process
        lang: Language code (e.g., 'ru-RU')
        model: Model name for classification. If None, uses default.
        
    Returns:
        PredictResponse with classification results or error message.
    """
    src_path: Optional[str] = None
    wav_path: Optional[str] = None
    
    try:
        # Validate file size
        data = await file.read()
        max_bytes = int(config.server.max_upload_mb * 1024 * 1024)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {config.server.max_upload_mb}MB"
            )
        
        # Save uploaded file
        suffix = os.path.splitext(file.filename or "audio")[1] or ".ogg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            src_path = tmp.name
        
        # Convert to WAV
        wav_path = audio_service.to_wav(src_path)
        if not wav_path:
            logger.error(f"Failed to convert audio to WAV: {file.filename}")
            raise HTTPException(
                status_code=415,
                detail="Unsupported audio format or FFmpeg error"
            )
        
        # Transcribe audio
        logger.info(
            f"Transcribing audio: {wav_path}, lang={lang}, "
            f"backend={audio_service.asr_backend}"
        )
        asr = audio_service.transcribe(wav_path, lang=lang)
        
        if not asr:
            return PredictResponse(
                success=False,
                error="Не удалось распознать речь. Проверьте настройки ASR_BACKEND в .env и качество аудио."
            )
        
        logger.info(
            f"ASR result: success=True, text_length={len(asr.text)}, "
            f"words={asr.word_count}, duration={asr.duration:.2f}s"
        )
        
        # Validate transcription result
        if not asr.text or not asr.text.strip():
            return PredictResponse(
                success=False,
                error="Не удалось распознать текст из аудио. Проверьте качество записи или настройки ASR.",
                duration=asr.duration,
                word_count=asr.word_count,
                asr_backend=asr.backend,
            )
        
        if audio_service.min_duration and asr.duration < audio_service.min_duration:
            return PredictResponse(
                success=False,
                error=f"Аудио слишком короткое (минимум {audio_service.min_duration} сек)",
                duration=asr.duration,
                word_count=asr.word_count,
                asr_backend=asr.backend,
            )
        
        if audio_service.min_words and asr.word_count < audio_service.min_words:
            return PredictResponse(
                success=False,
                error=f"Текст слишком короткий (минимум {audio_service.min_words} слов)",
                text=asr.text,
                duration=asr.duration,
                word_count=asr.word_count,
                asr_backend=asr.backend,
            )
        
        # Classify text
        model_name = model or config.model.default_model
        pred = text_classifier.predict(asr.text, model=model_name)
        
        if not pred.get("success"):
            return PredictResponse(
                success=False,
                error=pred.get("error", "Classification failed"),
                text=asr.text,
                duration=asr.duration,
                word_count=asr.word_count,
                asr_backend=asr.backend,
            )
        
        return PredictResponse(
            **pred,
            text=asr.text,
            duration=asr.duration,
            word_count=asr.word_count,
            asr_backend=asr.backend,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        # Clean up temporary files
        cleanup_files(src_path, wav_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )
