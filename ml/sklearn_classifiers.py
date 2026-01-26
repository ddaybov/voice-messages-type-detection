import pickle
from typing import Dict, Tuple

from .base_classifier import BaseClassifier


class SklearnClassifier(BaseClassifier):
    """Classifier based on sklearn models."""

    def __init__(
        self,
        name: str,
        model_path: str,
        vectorizer_path: str,
        description: str = "",
    ):
        super().__init__(name, description)
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None

    def load(self) -> None:
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_loaded = True

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.is_loaded:
            self.load()

        text = self.preprocess(text)
        X = self.vectorizer.transform([text])

        pred = self.model.predict(X)[0]
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("Model does not support predict_proba")
        proba = self.model.predict_proba(X)[0]

        label = "formal" if pred == 0 else "informal"
        confidence = float(max(proba))
        return label, confidence

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_loaded:
            self.load()

        text = self.preprocess(text)
        X = self.vectorizer.transform([text])
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("Model does not support predict_proba")
        proba = self.model.predict_proba(X)[0]

        return {"formal": float(proba[0]), "informal": float(proba[1])}
