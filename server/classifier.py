"""
DEPRECATED: Этот модуль не используется. Классификация выполняется через ml.model_factory
и классификаторы из пакета ml (server/main.py вызывает get_factory() и get_model()).
Оставлен для справки; можно удалить после проверки.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import joblib

from .config import config
from .constants import MODEL_ALIASES

# PyTorch imports - только при необходимости
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# Transformers imports - только при необходимости
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

logger = logging.getLogger(__name__)

# ----- DaybovNet (char-CNN, авторская) -----
# Определяем класс только если PyTorch доступен
if TORCH_AVAILABLE:
    class DaybovNet(nn.Module):
        def __init__(self, vocab_size:int, num_classes:int, embed_dim:int=64, num_filters:int=64, kernels=(3,5,7)):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernels])
            self.fc = nn.Linear(num_filters * len(kernels), num_classes)

        def forward(self, x):
            x = self.embedding(x).transpose(1,2)
            xs = [F.max_pool1d(F.relu(c(x)), c(x).shape[-1]).squeeze(-1) for c in self.convs]
            x = torch.cat(xs, dim=1)
            return self.fc(x)

    def encode_chars(text:str, vocab:dict, max_len:int=400):
        ids = [vocab.get(c.lower(), 0) for c in text[:max_len]]
        if len(ids) < max_len:
            ids += [0]*(max_len-len(ids))
        return torch.tensor(ids, dtype=torch.long)[None, :]
else:
    # Заглушки для случая, когда PyTorch не установлен
    class DaybovNet:
        pass
    
    def encode_chars(text:str, vocab:dict, max_len:int=400):
        raise ImportError("PyTorch not available")

@dataclass
class PredictResult:
    success: bool
    label: int | None = None
    label_name: str | None = None
    confidence: float | None = None
    text_length: int | None = None
    model: str | None = None
    error: str | None = None

class TextClassifier:
    """Registry + ensemble. Supports: sklearn, transformers, and DaybovNet."""
    
    def __init__(self, models_dir: str | None = None, model_config=None):
        """
        Initialize TextClassifier.
        
        Args:
            models_dir: Path to models directory. If None, uses config.
            model_config: ModelConfig instance. If None, uses global config.
        """
        self.config = model_config or config.model
        self.models_dir = models_dir or self.config.models_dir
        self.label_names: Dict[str, str] = {}
        self._hf: dict = {}   # name -> (tok, model)
        self._day_model: Optional[nn.Module] = None
        self._day_vocab: Optional[dict] = None
        self._cnn_vocab: Optional[dict] = None

    def available(self) -> Dict[str, str]:
        """Get dictionary of available models with descriptions."""
        return {
            "daybovnet": "Авторская char-CNN (DaybovNet)",
            "logreg": "Sklearn pipeline (Tfidf + LogisticRegression)",
            "svm": "Sklearn pipeline (Tfidf + LinearSVC)",
            "nb": "Sklearn pipeline (Tfidf + NaiveBayes)",
            "bert": "Transformer fine-tuned head",
            "dimanet": "Авторская CNN+BiLSTM+Attention",
            "ensemble": "Ансамбль (веса из models/ensemble.json)",
        }

    def load_labels(self):
        labels_path = os.path.join(self.models_dir, "labels.json")
        if os.path.exists(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.label_names = json.load(f)
            except Exception:
                logger.warning("labels.json exists but could not be parsed")

    def predict(self, text: str, model: str | None = None) -> Dict[str, Any]:
        """
        Predict text class using specified model.
        
        Args:
            text: Input text to classify
            model: Model name. If None, uses default from config.
            
        Returns:
            Dictionary with prediction results.
        """
        self.load_labels()
        model = model or self.config.default_model
        
        # Normalize model aliases
        model = MODEL_ALIASES.get(model.lower(), model)
        
        text = (text or "").strip()
        if not text:
            return PredictResult(
                False, 
                error="Не удалось распознать текст из аудио. Проверьте качество записи или настройки ASR."
            ).__dict__

        if model in ("logreg", "svm", "nb"):
            return self._predict_sklearn(text, model_id=model)
        if model == "bert":
            return self._predict_transformer(text)
        if model == "daybovnet":
            return self._predict_daybovnet(text)
        if model == "dimanet":
            return self._predict_dimanet(text)
        if model == "ensemble":
            return self._predict_ensemble(text)

        return PredictResult(
            False, 
            error=f"Неизвестная модель '{model}'. Доступные: daybovnet, logreg, svm, nb, bert, dimanet, ensemble"
        ).__dict__

    # ---- models ----
    def _predict_sklearn(self, text: str, model_id: str) -> Dict[str, Any]:
        path = os.path.join(self.models_dir, f"{model_id}.joblib")
        if not os.path.exists(path):
            return PredictResult(False, error=f"Model file not found: {path}").__dict__
        pipe = joblib.load(path)
        y_pred = pipe.predict([text])[0]
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba([text])[0]
            conf = float(np.max(proba)); label_idx = int(np.argmax(proba))
        else:
            label_idx = int(y_pred); conf = 1.0
        label_name = self.label_names.get(str(label_idx), str(label_idx))
        return PredictResult(True, label_idx, label_name, conf, len(text), model_id).__dict__

    def _ensure_daybovnet(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
        path = os.path.join(self.models_dir, "daybovnet.pt")
        if not os.path.exists(path):
            return False
        if self._day_model is None or self._day_vocab is None:
            state = torch.load(path, map_location="cpu")
            self._day_vocab = state["vocab"]
            num_classes = int(state.get("num_classes", 2))
            mdl = DaybovNet(vocab_size=len(self._day_vocab), num_classes=num_classes)
            sd = state["state_dict"] if "state_dict" in state else state
            mdl.load_state_dict(sd); mdl.eval()
            self._day_model = mdl
        return True

    def _predict_daybovnet(self, text: str) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return PredictResult(False, error="PyTorch not installed. DaybovNet requires PyTorch.").__dict__
        if not self._ensure_daybovnet():
            return PredictResult(False, error="daybovnet.pt not found in models/").__dict__
        x = encode_chars(text, self._day_vocab)
        with torch.no_grad():
            logits = self._day_model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        label_idx = int(np.argmax(probs)); conf = float(np.max(probs))
        label_name = self.label_names.get(str(label_idx), str(label_idx))
        return PredictResult(True, label_idx, label_name, conf, len(text), "daybovnet").__dict__

    def _predict_transformer(self, text: str) -> Dict[str, Any]:
        if not TRANSFORMERS_AVAILABLE:
            return PredictResult(False, error="Transformers library not installed").__dict__
        name = self.config.transformer_model
        if name not in self._hf:
            try:
                tok = AutoTokenizer.from_pretrained(name)
                mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=self.config.num_labels)
                self._hf[name] = (tok, mdl.eval())
            except Exception:
                logger.exception("Could not load transformer model: %s", name)
                return PredictResult(False, error="Transformer model not available").__dict__
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            return PredictResult(False, error="Transformers/PyTorch not installed").__dict__
        tok, mdl = self._hf[name]
        inputs = tok([text], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = mdl(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        label_idx = int(np.argmax(probs)); conf = float(np.max(probs))
        label_name = self.label_names.get(str(label_idx), str(label_idx))
        return PredictResult(True, label_idx, label_name, conf, len(text), "bert").__dict__

    # ---- ensemble ----
    def _predict_ensemble(self, text: str) -> Dict[str, Any]:
        members: List[tuple[str, np.ndarray]] = []
        weights: Dict[str,float] = {}

        spec = self.config.ensemble_weights
        for part in spec.split(","):
            if not part.strip(): continue
            name, w = part.split(":")
            weights[name.strip()] = float(w)

        if self._ensure_daybovnet() and weights.get("daybovnet",0)>0:
            x = encode_chars(text, self._day_vocab)
            with torch.no_grad():
                logits = self._day_model(x); p = F.softmax(logits, dim=-1).cpu().numpy()[0]
            members.append(("daybovnet", p))

        for m in ("logreg","svm","nb"):
            path = os.path.join(self.models_dir, f"{m}.joblib")
            if os.path.exists(path) and weights.get(m,0)>0:
                pipe = joblib.load(path)
                if hasattr(pipe, "predict_proba"):
                    p = pipe.predict_proba([text])[0]
                    members.append((m, p))

        if weights.get("bert",0)>0:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers unavailable for ensemble")
            else:
                name = self.config.transformer_model
                try:
                    tok, mdl = self._hf.get(name) or (None, None)
                    if tok is None:
                        tok = AutoTokenizer.from_pretrained(name)
                        mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=self.config.num_labels)
                        self._hf[name]=(tok, mdl.eval())
                    inputs = tok([text], return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        logits = mdl(**inputs).logits; p = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    members.append(("bert", p))
                except Exception:
                    logger.warning("Transformer unavailable for ensemble")

        if not members:
            return PredictResult(False, error="No models available for ensemble").__dict__

        probs = None; total_w = 0.0
        for name, p in members:
            w = float(weights.get(name, 1.0)); total_w += w
            probs = p*w if probs is None else probs + p*w
        probs = probs / max(total_w, 1e-9)
        label_idx = int(np.argmax(probs)); conf = float(np.max(probs))
        label_name = self.label_names.get(str(label_idx), str(label_idx))
        return PredictResult(True, label_idx, label_name, conf, len(text), f"ensemble[{','.join(n for n,_ in members)}]").__dict__


    def _predict_dimanet(self, text: str) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return PredictResult(False, error="PyTorch not installed. DimaNet requires PyTorch.").__dict__
        import json as _json
        vocab_path = os.path.join(self.models_dir, "dimanet_vocab.json")
        model_path = os.path.join(self.models_dir, "dimanet.pt")
        if not (os.path.exists(vocab_path) and os.path.exists(model_path)):
            return PredictResult(False, error="DimaNet artifacts not found (dimanet_vocab.json, dimanet.pt)").__dict__
        if self._cnn_vocab is None:
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self._cnn_vocab = _json.load(f)
            except Exception as e:
                logger.exception("Error loading DimaNet vocab: %s", e)
                return PredictResult(False, error=f"Error loading DimaNet vocabulary: {e}").__dict__
        state = torch.load(model_path, map_location="cpu")

        # Определяем DimaNet класс внутри функции (локально)
        class DimaNet(nn.Module):
            def __init__(self, vocab_size, emb_dim=128, lstm_dim=128, num_classes=2):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.dw = nn.ModuleList([nn.Conv1d(emb_dim, emb_dim, k, groups=emb_dim, padding=k//2) for k in (3,5)])
                self.pw = nn.ModuleList([nn.Conv1d(emb_dim, emb_dim//2, 1) for _ in (3,5)])
                self.bilstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_dim, bidirectional=True, batch_first=True)
                self.attn = nn.Linear(2*lstm_dim, 1)
                self.fc = nn.Linear(2*lstm_dim, num_classes)
            def forward(self, x):
                e = self.emb(x).transpose(1,2)
                convs = []
                for dw, pw in zip(self.dw, self.pw):
                    y = F.relu(pw(dw(e)))
                    convs.append(y)
                c = torch.cat(convs, dim=1).transpose(1,2)
                h,_ = self.bilstm(c)
                a = torch.softmax(self.attn(h).squeeze(-1), dim=-1)
                v = torch.sum(h * a.unsqueeze(-1), dim=1)
                return self.fc(v)

        lstm_dim = int(state.get("lstm_dim", 128))
        emb_dim = int(state.get("emb_dim", 128))
        num_classes = int(state.get("num_classes", 2))
        model = DimaNet(vocab_size=len(self._cnn_vocab)+1, emb_dim=emb_dim, lstm_dim=lstm_dim, num_classes=num_classes)
        sd = state.get("state_dict", state)
        model.load_state_dict(sd); model.eval()

        tokens = [self._cnn_vocab.get(tok, 0) for tok in text.lower().split()]
        if not tokens:
            return PredictResult(False, error="Не удалось распознать текст из аудио для DimaNet").__dict__
        x = torch.tensor(tokens, dtype=torch.long)[None,:]
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            label_idx = int(np.argmax(probs)); conf = float(np.max(probs))
        label_name = self.label_names.get(str(label_idx), str(label_idx))
        return PredictResult(True, label_idx, label_name, conf, len(text), "dimanet").__dict__

    def _predict_ensemble(self, text: str) -> Dict[str, Any]:
        """Weighted average of base learners from models/ensemble.json"""
        cfg_path = os.path.join(self.models_dir, "ensemble.json")
        if not os.path.exists(cfg_path):
            return {"success": False, "error": "ensemble.json not found in models/"}
        try:
            import json as _json
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = _json.load(f)
            members = cfg.get("members", [])
        except Exception:
            return {"success": False, "error": "Invalid ensemble.json"}

        probs = None
        used = 0.0
        num_labels = self.config.num_labels
        for m in members:
            mid = m.get("model")
            w = float(m.get("weight", 1.0))
            pr = self.predict(text, model=mid)
            if not pr.get("success"):
                continue
            vec = np.zeros(num_labels, dtype=float)
            vec[int(pr.get("label", 0))] = float(pr.get("confidence", 1.0))
            probs = vec*w if probs is None else probs + vec*w
            used += w
        if probs is None:
            return {"success": False, "error": "No base predictions available for ensemble."}
        label_idx = int(np.argmax(probs)); conf = float(np.max(probs) / max(used, 1e-9))
        label_name = self.label_names.get(str(label_idx), str(label_idx))
        return PredictResult(True, label_idx, label_name, conf, len(text), "ensemble").__dict__


# Global instance (initialized in main.py)
text_classifier = TextClassifier()
