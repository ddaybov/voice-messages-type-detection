
import os
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .audio_processor import audio_service
from .classifier import text_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

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

app = FastAPI(title="Speech Style Classifier", version="2.0.0")
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
    model: str = Form(None),
):
    suffix = os.path.splitext(file.filename or "audio")[1] or ".ogg"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await file.read()
            max_mb = float(os.getenv("MAX_UPLOAD_MB", "20"))
            if len(data) > max_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large")
            tmp.write(data)
            src_path = tmp.name
    except Exception as e:
        logger.exception("Could not store uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Upload failed") from e

    wav_path = audio_service.to_wav(src_path)
    if not wav_path:
        os.unlink(src_path)
        raise HTTPException(status_code=415, detail="Unsupported audio or FFmpeg error")
    asr = audio_service.transcribe(wav_path, lang=lang)
    for p in (src_path, wav_path):
        try: os.unlink(p)
        except Exception: pass
    if not asr:
        return PredictResponse(success=False, error="ASR failed (check backend and env)")
    if audio_service.min_duration and asr.duration < audio_service.min_duration:
        return PredictResponse(success=False, error=f"Audio shorter than MIN_DURATION={audio_service.min_duration}s")
    if audio_service.min_words and asr.word_count < audio_service.min_words:
        return PredictResponse(success=False, error=f"Text shorter than MIN_WORDS={audio_service.min_words}")

    pred = text_classifier.predict(asr.text, model=model or os.getenv("DEFAULT_MODEL", "daybovnet"))
    if not pred.get("success"):
        return PredictResponse(success=False, error=pred.get("error"), text=asr.text, duration=asr.duration, word_count=asr.word_count, asr_backend=asr.backend)

    return PredictResponse(
        **pred, text=asr.text, duration=asr.duration, word_count=asr.word_count, asr_backend=asr.backend
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8000")), log_level=os.getenv("LOG_LEVEL","info"))
