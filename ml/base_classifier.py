from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseClassifier(ABC):
    """Base class for all classifiers."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model artifacts."""

    @abstractmethod
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict class label.
        Returns:
            (label, confidence) - label and confidence.
        """

    @abstractmethod
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Class probabilities.
        Returns:
            {"formal": 0.8, "informal": 0.2}
        """

    def preprocess(self, text: str) -> str:
        """Basic text preprocessing."""
        import re

        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text
