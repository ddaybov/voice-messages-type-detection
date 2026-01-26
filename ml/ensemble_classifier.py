from typing import Dict, Tuple

from .base_classifier import BaseClassifier


class EnsembleClassifier(BaseClassifier):
    """Ensemble classifier."""

    def __init__(
        self,
        name: str,
        classifiers: Dict[str, BaseClassifier],
        weights: Dict[str, float] | None = None,
        description: str = "",
    ):
        super().__init__(name, description)
        self.classifiers = classifiers
        self.weights = weights or {k: 1.0 for k in classifiers.keys()}

        total = sum(self.weights.values()) or 1.0
        self.weights = {k: v / total for k, v in self.weights.items()}

    def load(self) -> None:
        for clf in self.classifiers.values():
            if not clf.is_loaded:
                clf.load()
        self.is_loaded = True

    def predict(self, text: str) -> Tuple[str, float]:
        proba = self.predict_proba(text)
        label = "formal" if proba["formal"] > proba["informal"] else "informal"
        confidence = max(proba.values())
        return label, confidence

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_loaded:
            self.load()

        ensemble_proba = {"formal": 0.0, "informal": 0.0}
        for name, clf in self.classifiers.items():
            proba = clf.predict_proba(text)
            weight = self.weights.get(name, 1.0)
            ensemble_proba["formal"] += weight * proba["formal"]
            ensemble_proba["informal"] += weight * proba["informal"]

        return ensemble_proba
