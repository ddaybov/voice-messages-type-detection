
import argparse, json, string
from pathlib import Path
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# -------- Model --------
class DaybovNet(nn.Module):
    \"\"\"Char-level CNN with multi-kernel conv and max-over-time pooling.\"\"\"
    def __init__(self, vocab_size:int, num_classes:int, embed_dim:int=64, num_filters:int=64, kernels=(3,5,7)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernels])
        self.fc = nn.Linear(num_filters * len(kernels), num_classes)

    def forward(self, x):
        # x: (B, T)
        x = self.embedding(x).transpose(1,2)      # (B, E, T)
        xs = [nn.functional.max_pool1d(nn.functional.relu(c(x)), c(x).shape[-1]).squeeze(-1) for c in self.convs]
        x = torch.cat(xs, dim=1)
        return self.fc(x)

# -------- Data --------
ALPHABET = list(" абвгдеёжзийклмнопрстуфхцчшщьыъэюя" + string.ascii_lowercase + string.digits + string.punctuation)
def build_vocab(texts):
    chars = set(c for t in texts for c in t.lower())
    vocab = {"<pad>":0}
    for ch in ALPHABET:
        if ch in chars and ch not in vocab:
            vocab[ch] = len(vocab)
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab

class CharDS(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=400):
        self.vocab = vocab or build_vocab(texts)
        self.max_len = max_len
        self.X = [self.encode(t) for t in texts]
        self.y = torch.tensor(labels, dtype=torch.long)
    def encode(self, t):
        ids = [self.vocab.get(c.lower(), 0) for c in t[:self.max_len]]
        if len(ids) < self.max_len:
            ids += [0]*(self.max_len-len(ids))
        return torch.tensor(ids, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train(args):
    df = pd.read_csv(args.data)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype('category').cat.codes.tolist()
    label_map = dict(enumerate(df['label'].astype('category').cat.categories))
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    ds_tr = CharDS(Xtr, ytr); ds_te = CharDS(Xte, yte, vocab=ds_tr.vocab)
    tr = DataLoader(ds_tr, batch_size=32, shuffle=True); te = DataLoader(ds_te, batch_size=128)

    model = DaybovNet(vocab_size=len(ds_tr.vocab), num_classes=len(set(labels)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(args.epochs):
        for xb, yb in tr:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()

    # Eval
    model.eval(); preds=[]; gts=[]
    with torch.no_grad():
        for xb, yb in te:
            logits = model(xb)
            preds += logits.argmax(dim=-1).cpu().tolist()
            gts += yb.cpu().tolist()
    macro = f1_score(gts, preds, average='macro')
    print("macro-f1:", macro)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": ds_tr.vocab, "num_classes": len(set(labels))}, outdir / "daybovnet.pt")
    with open(outdir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({str(k): str(v) for k,v in label_map.items()}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    train(args)
