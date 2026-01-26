from typing import Dict, Tuple

try:
    import torch
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    pipeline = None

from .base_classifier import BaseClassifier


class PretrainedClassifier(BaseClassifier):
    """Classifier based on pretrained HuggingFace models."""

    def __init__(self, name: str, model_id: str, description: str = ""):
        super().__init__(name, description)
        self.model_id = model_id
        self.pipe = None

    def load(self) -> None:
        if pipeline is None:
            raise ImportError("transformers is not installed")
        self.pipe = pipeline(
            "text-classification",
            model=self.model_id,
            device=0 if torch and torch.cuda.is_available() else -1,
        )
        self.is_loaded = True

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.is_loaded:
            self.load()

        result = self.pipe(text)[0]
        label = result["label"].lower()

        if label in ["formal", "informal"]:
            pass
        elif label in ["positive", "neutral"]:
            label = "formal"
        elif label == "negative":
            label = "informal"
        else:
            label = "formal" if "0" in label or "formal" in label else "informal"

        return label, float(result["score"])

    def predict_proba(self, text: str) -> Dict[str, float]:
        label, confidence = self.predict(text)
        if label == "formal":
            return {"formal": confidence, "informal": 1 - confidence}
        return {"formal": 1 - confidence, "informal": confidence}
