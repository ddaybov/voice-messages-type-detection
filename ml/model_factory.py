"""
Фабрика моделей для динамической загрузки классификаторов.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from .base_classifier import BaseClassifier
from .sklearn_classifiers import (
    LogisticRegressionClassifier,
    SVMClassifier,
    NaiveBayesClassifier,
    RandomForestClassifier,
    DaybovClassifier,
)
from .pytorch_classifiers import BiLSTMClassifier, CNNClassifier
from .bert_classifier import BertClassifier
from .ensemble_classifier import EnsembleClassifier


class ModelFactory:
    """Фабрика для создания и управления классификаторами"""

    MODEL_CLASSES = {
        "daybov": DaybovClassifier,
        "logreg": LogisticRegressionClassifier,
        "svm": SVMClassifier,
        "naive_bayes": NaiveBayesClassifier,
        "random_forest": RandomForestClassifier,
        "bilstm": BiLSTMClassifier,
        "cnn": CNNClassifier,
        "bert": BertClassifier,
    }

    MODEL_INFO = {
        "daybov": {
            "name": "Daybov Model",
            "description": "TF-IDF + Logistic Regression (авторская)",
            "emoji": "🎯",
            "type": "sklearn",
        },
        "logreg": {
            "name": "Logistic Regression",
            "description": "Классическая логистическая регрессия",
            "emoji": "📊",
            "type": "sklearn",
        },
        "svm": {
            "name": "SVM",
            "description": "Support Vector Machine",
            "emoji": "⚡",
            "type": "sklearn",
        },
        "naive_bayes": {
            "name": "Naive Bayes",
            "description": "Наивный байесовский классификатор",
            "emoji": "🎲",
            "type": "sklearn",
        },
        "random_forest": {
            "name": "Random Forest",
            "description": "Случайный лес",
            "emoji": "🌲",
            "type": "sklearn",
        },
        "bilstm": {
            "name": "BiLSTM",
            "description": "Bidirectional LSTM нейросеть",
            "emoji": "🔄",
            "type": "pytorch",
        },
        "cnn": {
            "name": "CNN",
            "description": "Сверточная нейросеть",
            "emoji": "🧠",
            "type": "pytorch",
        },
        "bert": {
            "name": "BERT",
            "description": "Трансформер rubert-tiny2",
            "emoji": "🤖",
            "type": "transformer",
        },
        "ensemble": {
            "name": "Ensemble",
            "description": "Ансамбль всех моделей",
            "emoji": "🎭",
            "type": "ensemble",
        },
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._cache: Dict[str, BaseClassifier] = {}
        self._load_config()

    def _load_config(self) -> None:
        config_path = self.models_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {"default": "daybov", "available": list(self.MODEL_CLASSES.keys())}

    def get_model(self, model_id: str) -> BaseClassifier:
        if model_id in self._cache:
            return self._cache[model_id]

        if model_id == "ensemble":
            ensemble_path = self.models_dir / "ensemble.json"
            classifier = EnsembleClassifier.from_config(str(ensemble_path), self)
        elif model_id in self.MODEL_CLASSES:
            classifier = self.MODEL_CLASSES[model_id](str(self.models_dir))
        else:
            raise ValueError(f"Неизвестная модель: {model_id}")

        self._cache[model_id] = classifier
        return classifier

    def get_available_models(self) -> Dict[str, dict]:
        available = {}
        for model_id, info in self.MODEL_INFO.items():
            if self._model_exists(model_id):
                available[model_id] = info.copy()
        return available

    def _model_exists(self, model_id: str) -> bool:
        if model_id == "ensemble":
            return (self.models_dir / "ensemble.json").exists()
        if model_id in ["daybov", "logreg"]:
            return (self.models_dir / "logreg.pkl").exists()
        if model_id == "svm":
            return (self.models_dir / "svm.pkl").exists()
        if model_id == "naive_bayes":
            return (self.models_dir / "naive_bayes.pkl").exists()
        if model_id == "random_forest":
            return (self.models_dir / "random_forest.pkl").exists()
        if model_id == "bilstm":
            return (self.models_dir / "bilstm.pt").exists()
        if model_id == "cnn":
            return (self.models_dir / "cnn.pt").exists()
        if model_id == "bert":
            return (self.models_dir / "bert").is_dir()
        return False

    def get_default_model(self) -> str:
        return self.config.get("default", "daybov")

    def preload_models(self, model_ids: list = None) -> None:
        if model_ids is None:
            model_ids = list(self.get_available_models().keys())
        print(f"📦 Предзагрузка {len(model_ids)} моделей...")
        for model_id in model_ids:
            try:
                model = self.get_model(model_id)
                model.load()
                print(f"  ✅ {model_id}")
            except Exception as e:
                print(f"  ⚠️ {model_id}: {e}")


_factory = None


def get_factory(models_dir: str = "models") -> ModelFactory:
    global _factory
    if _factory is None:
        _factory = ModelFactory(models_dir)
    return _factory
