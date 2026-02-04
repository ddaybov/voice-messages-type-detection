"""
Обучение sklearn моделей (LogReg, SVM, NB, RF).
Использует уже подготовленные данные из data/train.csv
"""

import pickle
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score


def train_all(data_path: str, models_dir: str = "models"):
    print(f"📥 Загрузка данных из {data_path}...")
    df = pd.read_csv(data_path)

    df["word_count"] = df["text"].str.split().str.len()
    print("\n📏 Распределение длин:")
    print(df.groupby("label")["word_count"].agg(["mean", "std"]).round(1))

    label_map = {"formal": 0, "informal": 1}
    df["label_encoded"] = df["label"].map(label_map)

    X = df["text"].str.lower().values
    y = df["label_encoded"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n🔄 TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"   Признаков: {X_train_tfidf.shape[1]}")

    with open(f"{models_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    models = {
        "logreg": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "svm": SVC(
            C=1.0,
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "naive_bayes": MultinomialNB(alpha=0.1),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n🔄 {name}...")
        model.fit(X_train_tfidf, y_train)

        cv_scores = cross_val_score(
            model, X_train_tfidf, y_train, cv=5, scoring="f1_macro"
        )
        print(f"   CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"   Test Accuracy: {acc:.4f}")
        print(f"   Test F1-macro: {f1:.4f}")

        if f1 > 0.95:
            print(
                f"   ⚠️ ВНИМАНИЕ: F1={f1:.4f} > 0.95 - проверьте распределение длин!"
            )

        results[name] = {"accuracy": acc, "f1": f1}

        with open(f"{models_dir}/{name}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"   ✅ Сохранено: {models_dir}/{name}.pkl")

    print("\n" + "=" * 50)
    print("📊 ИТОГИ")
    print("=" * 50)
    for name, metrics in results.items():
        print(f"   {name}: F1={metrics['f1']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv", help="Путь к подготовленным данным")
    parser.add_argument("--models-dir", default="models", help="Директория для моделей")
    args = parser.parse_args()

    train_all(args.data, args.models_dir)


if __name__ == "__main__":
    main()
