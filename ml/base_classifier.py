"""
Абстрактный базовый класс для всех классификаторов.
Обеспечивает единый интерфейс для всех моделей.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict
import re


class BaseClassifier(ABC):
    """Абстрактный класс для классификаторов formal/informal"""

    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.label_map = {0: "formal", 1: "informal"}
        self.label_map_inv = {"formal": 0, "informal": 1}

    @abstractmethod
    def load(self) -> None:
        """Загрузить модель с диска"""

    @abstractmethod
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Предсказать класс текста.

        Args:
            text: Входной текст

        Returns:
            (label, confidence): Метка класса и уверенность
        """

    @abstractmethod
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Получить вероятности классов.

        Args:
            text: Входной текст

        Returns:
            {'formal': prob, 'informal': prob}
        """

    def preprocess(self, text: str) -> str:
        """Предобработка текста"""
        text = str(text).lower().strip()
        text = re.sub(r"http\\S+|www\\S+", "", text)
        text = re.sub(r"@\\w+", "", text)
        text = re.sub(r"#\\w+", "", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def ensure_loaded(self) -> None:
        """Убедиться что модель загружена"""
        if not self.is_loaded:
            self.load()
