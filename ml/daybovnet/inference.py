
import json, torch, torch.nn as nn, torch.nn.functional as F

class DaybovNet(nn.Module):
    def __init__(self, vocab_size:int, num_classes:int, embed_dim:int=64, num_filters:int=64, kernels=(3,5,7)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernels])
        self.fc = nn.Linear(num_filters * len(kernels), num_classes)
    def forward(self, x):
        x = self.embedding(x).transpose(1,2)
        xs = [nn.functional.max_pool1d(nn.functional.relu(c(x)), c(x).shape[-1]).squeeze(-1) for c in self.convs]
        x = torch.cat(xs, dim=1)
        return self.fc(x)

def load_daybovnet(model_path:str):
    state = torch.load(model_path, map_location="cpu")
    vocab = state["vocab"]; num_classes = int(state.get("num_classes", 2))
    model = DaybovNet(vocab_size=len(vocab), num_classes=num_classes)
    sd = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(sd); model.eval()
    return model, vocab

def encode_text(text:str, vocab:dict, max_len:int=400):
    ids = [vocab.get(c.lower(), 0) for c in text[:max_len]]
    if len(ids) < max_len: ids += [0]*(max_len-len(ids))
    return torch.tensor(ids, dtype=torch.long)[None, :]

def predict_daybovnet(text:str, model, vocab):
    x = encode_text(text, vocab)
    with torch.no_grad():
        logits = model(x); probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    pred = int(probs.argmax()); conf = float(probs.max())
    return pred, conf
