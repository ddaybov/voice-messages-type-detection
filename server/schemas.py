"""
Pydantic-схемы ответов API.
"""

from typing import Optional
from pydantic import BaseModel


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


class PredictTextResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None
    model: Optional[str] = None
    error: Optional[str] = None
