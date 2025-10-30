
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

def get_model_predict_fn(model_id: str, models_dir: Path):
    if model_id in ('logreg','svm','nb'):
        pipe = joblib.load(models_dir/f"{model_id}.joblib")
        def _pred(texts):
            yhat = pipe.predict(texts)
            conf = pipe.predict_proba(texts) if hasattr(pipe,'predict_proba') else None
            if conf is None:
                # One-hot fallback
                n = len(set(yhat))
                conf = np.eye(n)[yhat]
            return yhat, conf
        return _pred
    raise NotImplementedError(f"{model_id} ensemble training not implemented")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--models_dir', default=Path('models'), type=Path)
    ap.add_argument('--members', nargs='+', default=['logreg'])
    ap.add_argument('--out', default=Path('models/ensemble.json'), type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    texts = df['text'].astype(str).tolist()
    y = df['label'].astype('category').cat.codes.to_numpy()

    prob_features = []
    for mid in args.members:
        pred_fn = get_model_predict_fn(mid, args.models_dir)
        _, prob = pred_fn(texts)
        prob_features.append(prob)
    X = np.concatenate(prob_features, axis=1)

    meta = LogisticRegression(max_iter=200)
    meta.fit(X, y)
    yhat = meta.predict(X)
    print('macro-F1 (train):', f1_score(y, yhat, average='macro'))

    coeff = np.abs(meta.coef_).mean(axis=0)
    k = coeff.shape[0] // len(args.members)
    weights = []
    for i, mid in enumerate(args.members):
        w = float(coeff[i*k:(i+1)*k].sum())
        weights.append({'model': mid, 'weight': round(w, 6)})
    s = sum(w['weight'] for w in weights) or 1.0
    for w in weights:
        w['weight'] = round(w['weight']/s, 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'members': weights}, f, ensure_ascii=False, indent=2)
    print('Saved ensemble:', args.out, weights)

if __name__ == '__main__':
    main()
