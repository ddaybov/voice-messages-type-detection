import json
from typing import Dict, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from .base_classifier import BaseClassifier


class BiLSTMModel(nn.Module):
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
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        filter_sizes=None,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        filter_sizes = filter_sizes or [3, 4, 5]
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
    """Classifier based on PyTorch models."""

    def __init__(
        self,
        name: str,
        model_path: str,
        vocab_path: str,
        model_class: str,
        description: str = "",
        max_len: int = 64,
    ):
        super().__init__(name, description)
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model_class = model_class
        self.max_len = max_len
        self.model = None
        self.vocab = None
        self.device = None

    def load(self) -> None:
        if torch is None:
            raise ImportError("PyTorch is not installed")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        if self.model_class == "bilstm":
            self.model = BiLSTMModel(len(self.vocab))
        elif self.model_class == "cnn":
            self.model = CNNModel(len(self.vocab))
        else:
            raise ValueError(f"Unknown model_class: {self.model_class}")

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def _tokenize(self, text: str) -> "torch.Tensor":
        words = text.split()
        unk_id = self.vocab.get("<UNK>", 1)
        pad_id = self.vocab.get("<PAD>", 0)
        indices = [self.vocab.get(w, unk_id) for w in words]

        if len(indices) < self.max_len:
            indices += [pad_id] * (self.max_len - len(indices))
        else:
            indices = indices[: self.max_len]

        return torch.tensor([indices], dtype=torch.long).to(self.device)

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.is_loaded:
            self.load()

        text = self.preprocess(text)
        input_ids = self._tokenize(text)

        with torch.no_grad():
            outputs = self.model(input_ids)
            proba = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(proba).item()

        label = "formal" if pred == 0 else "informal"
        confidence = float(proba[pred])
        return label, confidence

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_loaded:
            self.load()

        text = self.preprocess(text)
        input_ids = self._tokenize(text)

        with torch.no_grad():
            outputs = self.model(input_ids)
            proba = torch.softmax(outputs, dim=1)[0]

        return {"formal": float(proba[0]), "informal": float(proba[1])}
