"""
Обучение BERT модели.
Использует подготовленные данные из data/train.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--output", default="models/bert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print(f"📥 Загрузка {args.data}...")
    df = pd.read_csv(args.data)

    df["word_count"] = df["text"].str.split().str.len()
    print("\n📏 Распределение длин:")
    print(df.groupby("label")["word_count"].agg(["mean", "std"]).round(1))

    df["label"] = df["label"].map({"formal": 0, "informal": 1})

    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.176, random_state=42, stratify=train_df["label"]
    )

    print(f"\n📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    model_name = "cointegrated/rubert-tiny2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]]).map(
        tokenize, batched=True
    )
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]]).map(
        tokenize, batched=True
    )
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]]).map(
        tokenize, batched=True
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds, average="macro")

        if f1 > 0.95:
            print(f"\n⚠️ F1={f1:.4f} > 0.95 - возможно переобучение на длину!")

        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1,
        }

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n🔄 Обучение BERT...")
    trainer.train()

    results = trainer.evaluate(test_dataset)
    print("\n📊 Результаты:")
    print(f"   Accuracy: {results['eval_accuracy']:.4f}")
    print(f"   F1-macro: {results['eval_macro_f1']:.4f}")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n✅ Модель: {args.output}")


if __name__ == "__main__":
    main()
