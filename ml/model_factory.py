import json
import os
from typing import Dict, Optional

from .base_classifier import BaseClassifier
from .sklearn_classifiers import SklearnClassifier
from .pytorch_classifiers import PyTorchClassifier
from .bert_classifier import BertClassifier
from .pretrained_classifier import PretrainedClassifier
from .ensemble_classifier import EnsembleClassifier


class ModelFactory:
    """Factory for creating and managing models."""

    def __init__(self, config_path: str = "models/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.models: Dict[str, BaseClassifier] = {}

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Model config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_available_models(self) -> Dict[str, dict]:
        result = {}
        for category in ["trained", "pretrained"]:
            if category in self.config["models"]:
                for model_id, model_info in self.config["models"][category].items():
                    result[model_id] = {
                        "name": model_info["name"],
                        "description": model_info.get("description", ""),
                        "emoji": model_info.get("emoji", "🤖"),
                        "category": category,
                    }
        return result

    def get_model(self, model_id: str) -> BaseClassifier:
        if model_id in self.models:
            return self.models[model_id]

        if model_id in self.config["models"].get("trained", {}):
            model_info = self.config["models"]["trained"][model_id]
            model = self._create_trained_model(model_id, model_info)
        elif model_id in self.config["models"].get("pretrained", {}):
            model_info = self.config["models"]["pretrained"][model_id]
            model = self._create_pretrained_model(model_id, model_info)
        else:
            raise ValueError(f"Unknown model: {model_id}")

        self.models[model_id] = model
        return model

    def _create_trained_model(self, model_id: str, info: dict) -> BaseClassifier:
        model_type = info["type"]

        if model_type == "sklearn":
            return SklearnClassifier(
                name=info["name"],
                model_path=info["path"],
                vectorizer_path=info["vectorizer"],
                description=info.get("description", ""),
            )
        if model_type == "pytorch":
            return PyTorchClassifier(
                name=info["name"],
                model_path=info["path"],
                vocab_path=info["vocab"],
                model_class=model_id,
                description=info.get("description", ""),
            )
        if model_type == "transformers":
            return BertClassifier(
                name=info["name"],
                model_path=info["path"],
                description=info.get("description", ""),
            )
        if model_type == "ensemble":
            return self._create_ensemble(info)

        raise ValueError(f"Unknown model type: {model_type}")

    def _create_pretrained_model(self, model_id: str, info: dict) -> BaseClassifier:
        return PretrainedClassifier(
            name=info["name"],
            model_id=info["model_id"],
            description=info.get("description", ""),
        )

    def _create_ensemble(self, info: dict) -> EnsembleClassifier:
        config_path = info["config"]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Ensemble config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            ensemble_config = json.load(f)

        classifiers = {}
        for model_id in ensemble_config["models"]:
            classifiers[model_id] = self.get_model(model_id)

        return EnsembleClassifier(
            name=info["name"],
            classifiers=classifiers,
            weights=ensemble_config.get("weights"),
            description=info.get("description", ""),
        )

    def get_default_model(self) -> BaseClassifier:
        default_id = self.config.get("default_model", "logreg")
        return self.get_model(default_id)


_factory: Optional[ModelFactory] = None


def get_factory() -> ModelFactory:
    global _factory
    if _factory is None:
        _factory = ModelFactory()
    return _factory
