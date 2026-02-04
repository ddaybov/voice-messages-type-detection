"""
Ансамбль классификаторов с взвешенным голосованием.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from .base_classifier import BaseClassifier


class EnsembleClassifier(BaseClassifier):
    """Ансамбль классификаторов"""

    def __init__(self, classifiers: Dict[str, BaseClassifier], weights: Dict[str, float] = None):
        super().__init__()
        self.classifiers = classifiers
        self.weights = weights or {k: 1.0 for k in classifiers.keys()}

        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def load(self) -> None:
        """Загрузить все модели ансамбля"""
        for name, clf in self.classifiers.items():
            try:
                clf.load()
                print(f"  ✅ {name} загружен")
            except Exception as e:
                print(f"  ⚠️ {name} не загружен: {e}")
        self.is_loaded = True

    def predict(self, text: str) -> Tuple[str, float]:
        """Взвешенное предсказание"""
        proba = self.predict_proba(text)

        if proba["formal"] > proba["informal"]:
            return "formal", proba["formal"]
        return "informal", proba["informal"]

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Взвешенное усреднение вероятностей"""
        self.ensure_loaded()

        formal_prob = 0.0
        informal_prob = 0.0

        for name, clf in self.classifiers.items():
            if clf.is_loaded:
                try:
                    proba = clf.predict_proba(text)
                    weight = self.weights.get(name, 0)
                    formal_prob += weight * proba["formal"]
                    informal_prob += weight * proba["informal"]
                except Exception as e:
                    print(f"⚠️ {name} ошибка предсказания: {e}")

        total = formal_prob + informal_prob
        if total > 0:
            formal_prob /= total
            informal_prob /= total

        return {"formal": formal_prob, "informal": informal_prob}

    @classmethod
    def from_config(cls, config_path: str, model_factory) -> "EnsembleClassifier":
        """Создать ансамбль из конфигурации"""
        with open(config_path, "r") as f:
            config = json.load(f)

        classifiers = {}
        weights: Dict[str, float] = {}

        # New format: {"members": [{"model": "logreg", "weight": 0.5}, ...]}
        if isinstance(config, dict) and "members" in config:
            for member in config.get("members", []):
                model_id = member.get("model")
                if not model_id:
                    continue
                try:
                    classifiers[model_id] = model_factory.get_model(model_id)
                    if "weight" in member:
                        weights[model_id] = float(member["weight"])
                except Exception as e:
                    print(f"⚠️ Не удалось создать {model_id}: {e}")
        else:
            # Legacy format: {"models": [...], "weights": {...}}
            for model_id in config.get("models", []):
                try:
                    classifiers[model_id] = model_factory.get_model(model_id)
                except Exception as e:
                    print(f"⚠️ Не удалось создать {model_id}: {e}")
            weights = config.get("weights", {})

        return cls(classifiers=classifiers, weights=weights)
