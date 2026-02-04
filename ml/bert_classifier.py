"""
BERT классификатор для formal/informal классификации.
Использует предобученную rubert-tiny2 модель.
"""

import torch
from pathlib import Path
from typing import Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base_classifier import BaseClassifier


class BertClassifier(BaseClassifier):
    """BERT классификатор (rubert-tiny2)"""

    def __init__(self, model_path: str = "models/bert", max_length: int = 128):
        super().__init__()
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        """Загрузить модель и токенизатор"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def predict(self, text: str) -> Tuple[str, float]:
        """Предсказание с уверенностью"""
        self.ensure_loaded()

        text = self.preprocess(text)

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

        return self.label_map[pred], float(confidence)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Вероятности классов"""
        self.ensure_loaded()

        text = self.preprocess(text)

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        return {"formal": float(probs[0]), "informal": float(probs[1])}
