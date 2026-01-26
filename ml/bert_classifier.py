from typing import Dict, Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

from .base_classifier import BaseClassifier


class BertClassifier(BaseClassifier):
    """Classifier based on BERT."""

    def __init__(self, name: str, model_path: str, description: str = ""):
        super().__init__(name, description)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self) -> None:
        if torch is None or AutoTokenizer is None:
            raise ImportError("transformers is not installed")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.is_loaded:
            self.load()

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            proba = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(proba).item()

        label = "formal" if pred == 0 else "informal"
        confidence = float(proba[pred])
        return label, confidence

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_loaded:
            self.load()

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            proba = torch.softmax(outputs.logits, dim=1)[0]

        return {"formal": float(proba[0]), "informal": float(proba[1])}
