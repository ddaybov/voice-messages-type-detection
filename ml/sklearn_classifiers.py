"""
Sklearn классификаторы: LogReg, SVM, Naive Bayes, Random Forest.
Все используют TF-IDF векторизацию.
"""

import pickle
from pathlib import Path
from typing import Tuple, Dict
from .base_classifier import BaseClassifier


class SklearnClassifier(BaseClassifier):
    """Базовый класс для sklearn моделей"""

    def __init__(self, model_path: str, vectorizer_path: str):
        super().__init__()
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.vectorizer = None

    def load(self) -> None:
        """Загрузить модель и векторизатор"""
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_loaded = True

    def predict(self, text: str) -> Tuple[str, float]:
        """Предсказание с уверенностью"""
        self.ensure_loaded()
        text = self.preprocess(text)
        X = self.vectorizer.transform([text])

        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = proba[pred]

        return self.label_map[pred], float(confidence)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Вероятности классов"""
        self.ensure_loaded()
        text = self.preprocess(text)
        X = self.vectorizer.transform([text])

        proba = self.model.predict_proba(X)[0]
        return {"formal": float(proba[0]), "informal": float(proba[1])}


class LogisticRegressionClassifier(SklearnClassifier):
    """Logistic Regression классификатор"""

    def __init__(self, models_dir: str = "models"):
        super().__init__(
            model_path=f"{models_dir}/logreg.pkl",
            vectorizer_path=f"{models_dir}/tfidf_vectorizer.pkl",
        )


class SVMClassifier(SklearnClassifier):
    """SVM классификатор"""

    def __init__(self, models_dir: str = "models"):
        super().__init__(
            model_path=f"{models_dir}/svm.pkl",
            vectorizer_path=f"{models_dir}/tfidf_vectorizer.pkl",
        )


class NaiveBayesClassifier(SklearnClassifier):
    """Naive Bayes классификатор"""

    def __init__(self, models_dir: str = "models"):
        super().__init__(
            model_path=f"{models_dir}/naive_bayes.pkl",
            vectorizer_path=f"{models_dir}/tfidf_vectorizer.pkl",
        )


class RandomForestClassifier(SklearnClassifier):
    """Random Forest классификатор"""

    def __init__(self, models_dir: str = "models"):
        super().__init__(
            model_path=f"{models_dir}/random_forest.pkl",
            vectorizer_path=f"{models_dir}/tfidf_vectorizer.pkl",
        )


class DaybovClassifier(LogisticRegressionClassifier):
    """Авторская модель Daybov (TF-IDF + LogReg)"""

    pass
