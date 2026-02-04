"""
PyTorch классификаторы: BiLSTM, CNN.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict
from .base_classifier import BaseClassifier


class BiLSTMModel(nn.Module):
    """BiLSTM модель для классификации текста"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))


class CNNModel(nn.Module):
    """CNN модель для классификации текста"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: list = [3, 4, 5],
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conv_outputs = [torch.relu(conv(embedded)).max(dim=2)[0] for conv in self.convs]
        concatenated = torch.cat(conv_outputs, dim=1)
        return self.fc(self.dropout(concatenated))


class PyTorchClassifier(BaseClassifier):
    """Базовый класс для PyTorch моделей"""

    def __init__(self, model_path: str, vocab_path: str, max_len: int = 64):
        super().__init__()
        self.model_path = Path(model_path)
        self.vocab_path = Path(vocab_path)
        self.max_len = max_len
        self.vocab = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_vocab(self) -> None:
        """Загрузить словарь"""
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def tokenize(self, text: str) -> torch.Tensor:
        """Токенизация текста"""
        text = self.preprocess(text)
        words = text.split()

        indices = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]

        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[: self.max_len]

        return torch.tensor([indices], dtype=torch.long)

    def predict(self, text: str) -> Tuple[str, float]:
        """Предсказание с уверенностью"""
        self.ensure_loaded()

        input_ids = self.tokenize(text).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

        return self.label_map[pred], float(confidence)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Вероятности классов"""
        self.ensure_loaded()

        input_ids = self.tokenize(text).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids)
            probs = torch.softmax(outputs, dim=1)[0]

        return {"formal": float(probs[0]), "informal": float(probs[1])}


class BiLSTMClassifier(PyTorchClassifier):
    """BiLSTM классификатор"""

    def __init__(self, models_dir: str = "models"):
        super().__init__(
            model_path=f"{models_dir}/bilstm.pt",
            vocab_path=f"{models_dir}/vocab.json",
        )

    def load(self) -> None:
        """Загрузить модель"""
        self.load_vocab()

        self.model = BiLSTMModel(vocab_size=len(self.vocab))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True


class CNNClassifier(PyTorchClassifier):
    """CNN классификатор"""

    def __init__(self, models_dir: str = "models"):
        super().__init__(
            model_path=f"{models_dir}/cnn.pt",
            vocab_path=f"{models_dir}/vocab.json",
        )

    def load(self) -> None:
        """Загрузить модель"""
        self.load_vocab()

        self.model = CNNModel(vocab_size=len(self.vocab))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
