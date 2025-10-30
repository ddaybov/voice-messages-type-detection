
import argparse, json
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class DimaNet(nn.Module):
    """
    Авторская гибридная архитектура: depthwise-separable CNN → BiLSTM → attention pooling → FC.
    Лёгкая и быстрая, хорошо работает на коротких сообщениях.
    """
    def __init__(self, vocab_size, emb_dim=128, lstm_dim=128, num_classes=2, conv_k=(3,5)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # depthwise-separable conv
        self.dw = nn.ModuleList([nn.Conv1d(emb_dim, emb_dim, k, groups=emb_dim, padding=k//2) for k in conv_k])
        self.pw = nn.ModuleList([nn.Conv1d(emb_dim, emb_dim//2, 1) for _ in conv_k])
        self.bilstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_dim, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(2*lstm_dim, 1)
        self.fc = nn.Linear(2*lstm_dim, num_classes)

    def forward(self, x):
        # x: (B,T)
        e = self.emb(x).transpose(1,2)   # (B,E,T)
        convs = []
        for dw, pw in zip(self.dw, self.pw):
            y = F.relu(pw(dw(e)))        # (B,E2,T)
            convs.append(y)
        c = torch.cat(convs, dim=1).transpose(1,2)  # (B,T,E')
        h,_ = self.bilstm(c)             # (B,T,2H)
        # attention pooling
        a = torch.softmax(self.attn(h).squeeze(-1), dim=-1)  # (B,T)
        v = torch.sum(h * a.unsqueeze(-1), dim=1)            # (B,2H)
        return self.fc(v)

class TextDS(Dataset):
    def __init__(self, texts, labels, vocab=None, min_freq=2, max_len=256):
        toks = [t.lower().split() for t in texts]
        if vocab is None:
            cnt = Counter([w for ts in toks for w in ts])
            vocab = {w:i+1 for i,(w,c) in enumerate(cnt.items()) if c >= min_freq}
        self.vocab = vocab
        self.X = [torch.tensor([vocab.get(w,0) for w in ts][:max_len], dtype=torch.long) for ts in toks]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def collate(batch):
    xs, ys = zip(*batch)
    m = max(x.size(0) for x in xs)
    px = torch.stack([torch.cat([x, torch.zeros(m-x.size(0), dtype=torch.long)]) for x in xs])
    return px, torch.stack(ys)

def train_epoch(model, dl, opt, loss_fn):
    model.train()
    tot=0.0
    for xb,yb in dl:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        tot += float(loss)
    return tot / max(1,len(dl))

def evaluate(model, dl):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb,yb in dl:
            logits = model(xb)
            preds += logits.argmax(dim=-1).cpu().tolist()
            gts += yb.cpu().tolist()
    return f1_score(gts, preds, average='macro')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--outdir', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--emb_dim', type=int, default=128)
    ap.add_argument('--lstm_dim', type=int, default=128)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype('category').cat.codes.tolist()
    label_map = dict(enumerate(df['label'].astype('category').cat.categories))

    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    ds_tr = TextDS(Xtr, ytr); ds_te = TextDS(Xte, yte, vocab=ds_tr.vocab)
    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, collate_fn=collate)
    dl_te = DataLoader(ds_te, batch_size=64, collate_fn=collate)

    model = DimaNet(vocab_size=len(ds_tr.vocab)+1, emb_dim=args.emb_dim, lstm_dim=args.lstm_dim, num_classes=len(set(labels)))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        loss = train_epoch(model, dl_tr, opt, loss_fn)
        mf1 = evaluate(model, dl_te)
        print(f'Epoch {ep+1}: loss={loss:.4f} macro-F1={mf1:.4f}')

    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(set(labels)),
                "emb_dim": args.emb_dim, "lstm_dim": args.lstm_dim}, args.outdir/'dimanet.pt')
    with open(args.outdir/'dimanet_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(ds_tr.vocab, f, ensure_ascii=False)
    with open(args.outdir/'labels.json', 'w', encoding='utf-8') as f:
        json.dump({str(k): str(v) for k,v in label_map.items()}, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
