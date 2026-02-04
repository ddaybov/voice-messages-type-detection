"""
Подготовка данных для обучения моделей классификации formal/informal.

КРИТИЧЕСКИЕ ТЕХНИКИ:
1. Выравнивание длины текстов между классами (length normalization)
2. Добавление шума (noise injection)
3. Правильная очистка без потери семантики

Без этих техник модели могут достичь "идеальной" точности,
обучившись на различиях в длине текстов, а не на содержании!
"""

import os
import re
import random
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ datasets не установлен. pip install datasets")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def clean_text(text: str) -> str:
    """Очистка текста без потери семантики"""
    text = str(text).strip()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"\\S+@\\S+", "", text)
    text = re.sub(r"[@#]\\w+", "", text)
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"([!?.]){2,}", r"\\1", text)
    return text.strip()


def is_valid_text(text: str, min_words: int = 4, max_words: int = 100) -> bool:
    """Проверка валидности текста"""
    if not text:
        return False
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if not re.search(r"[а-яёА-ЯЁ]", text):
        return False
    return True


def normalize_lengths(
    formal_texts: List[str],
    informal_texts: List[str],
    target_range: Tuple[int, int] = (8, 40),
) -> Tuple[List[str], List[str]]:
    """Выравнивание распределения длин текстов между классами."""
    min_words, max_words = target_range

    formal_filtered = [
        t for t in formal_texts if min_words <= len(t.split()) <= max_words
    ]
    informal_filtered = [
        t for t in informal_texts if min_words <= len(t.split()) <= max_words
    ]

    print(f"\n📏 Выравнивание длины ({min_words}-{max_words} слов):")
    print(f"   Formal: {len(formal_texts)} → {len(formal_filtered)}")
    print(f"   Informal: {len(informal_texts)} → {len(informal_filtered)}")

    def bin_by_length(texts: List[str], n_bins: int = 10) -> dict:
        bins = {}
        for t in texts:
            word_count = len(t.split())
            bin_idx = min(word_count // 5, n_bins - 1)
            bins.setdefault(bin_idx, []).append(t)
        return bins

    formal_bins = bin_by_length(formal_filtered)
    informal_bins = bin_by_length(informal_filtered)

    formal_balanced = []
    informal_balanced = []
    all_bins = set(formal_bins.keys()) | set(informal_bins.keys())

    for bin_idx in all_bins:
        f_texts = formal_bins.get(bin_idx, [])
        i_texts = informal_bins.get(bin_idx, [])
        if f_texts and i_texts:
            n = min(len(f_texts), len(i_texts))
            formal_balanced.extend(random.sample(f_texts, n))
            informal_balanced.extend(random.sample(i_texts, n))

    print(f"   После балансировки: {len(formal_balanced)}, {len(informal_balanced)}")
    return formal_balanced, informal_balanced


def add_noise(text: str, noise_prob: float = 0.1) -> str:
    """Добавление шума к тексту для предотвращения переобучения."""
    if random.random() > noise_prob:
        return text

    words = text.split()
    if len(words) < 3:
        return text

    noise_type = random.choice(["drop", "case", "swap", "typo"])

    if noise_type == "drop" and len(words) > 4:
        idx = random.randint(1, len(words) - 2)
        words.pop(idx)
    elif noise_type == "case":
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx].lower() if random.random() > 0.5 else words[idx].upper()
    elif noise_type == "swap" and len(words) > 2:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    elif noise_type == "typo":
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 3:
            char_idx = random.randint(1, len(word) - 2)
            if random.random() > 0.5:
                word = word[:char_idx] + word[char_idx] + word[char_idx:]
            else:
                word = word[:char_idx] + word[char_idx + 1:]
            words[idx] = word

    return " ".join(words)


def augment_with_noise(texts: List[str], augment_factor: float = 0.3) -> List[str]:
    """Аугментация датасета с помощью зашумления"""
    augmented = list(texts)
    n_augment = int(len(texts) * augment_factor)

    for _ in range(n_augment):
        original = random.choice(texts)
        noisy = add_noise(original, noise_prob=0.8)
        augmented.append(noisy)

    print(f"   Аугментация: {len(texts)} → {len(augmented)}")
    return augmented


def load_formal_data(max_samples: int = 5000) -> List[str]:
    """Загрузка формальных текстов (новости)"""
    texts = []
    if not HF_AVAILABLE:
        return texts

    print("📥 Загрузка FORMAL данных...")
    try:
        print("   Lenta.ru...")
        lenta = load_dataset("IlyaGusev/lenta-ru-news", split="train", trust_remote_code=True)
        for item in list(lenta)[: max_samples * 2]:
            text = item.get("text", "")
            if not text:
                continue
            sentences = re.split(r"(?<=[.!?])\\s+", text)
            short_text = " ".join(sentences[:2]).strip()
            if short_text and not short_text.endswith((".", "!", "?")):
                short_text += "."
            short_text = clean_text(short_text)
            if is_valid_text(short_text):
                texts.append(short_text)
        print(f"   ✅ {len(texts)} текстов")
    except Exception as e:
        print(f"   ⚠️ Ошибка: {e}")

    return list(set(texts))[:max_samples]


def load_informal_data(max_samples: int = 5000) -> List[str]:
    """Загрузка неформальных текстов (соцсети)"""
    texts = []
    if not HF_AVAILABLE:
        return texts

    print("📥 Загрузка INFORMAL данных...")
    try:
        print("   RuSentiment (VK)...")
        rusentiment = load_dataset("RuSentiment/rusentiment", split="train", trust_remote_code=True)
        for item in rusentiment:
            text = item.get("text", "")
            text = clean_text(text)
            if is_valid_text(text, min_words=3, max_words=50):
                texts.append(text)
        print(f"   ✅ {len(texts)} текстов")
    except Exception as e:
        print(f"   ⚠️ Ошибка: {e}")

    return list(set(texts))[:max_samples]


FORMAL_TEMPLATES = [
    "Уважаемые коллеги, направляю вам отчёт о проделанной работе за текущий период.",
    "В соответствии с достигнутыми договорённостями, прошу подтвердить готовность к встрече.",
    "Настоящим уведомляем вас о необходимости предоставления запрашиваемых документов.",
    "Благодарим за оперативное решение вопроса и надеемся на дальнейшее сотрудничество.",
    "По результатам рассмотрения вашего обращения сообщаем следующее.",
    "Просим принять к сведению изменения в графике работы отдела.",
    "В связи с производственной необходимостью переносим совещание на следующую неделю.",
    "Направляю на согласование проект договора с указанными правками.",
    "По данным пресс-службы, мероприятие состоится в конце текущего месяца.",
    "Согласно официальной статистике, показатели выросли на пятнадцать процентов.",
]

INFORMAL_TEMPLATES = [
    "Привет! Как сам? Давно не виделись, соскучился уже!",
    "Здарова! Чё делаешь сегодня вечером? Может погуляем?",
    "Ой, привет! Слушай, хотела спросить тебя кое о чём.",
    "Хей! Ты видел что вчера было? Вообще жесть какая-то!",
    "Ахахах, это было так смешно, я чуть не умер со смеху!",
    "Блин, ну вот опять! Достало уже это всё, честное слово.",
    "Офигеть! Серьёзно? Не могу поверить что это правда!",
    "Капец какой-то, у меня просто слов нет от этого всего.",
    "Слушай, можешь скинуть те фотки с выходных? Плиз!",
    "Братан, выручай, срочно нужна твоя помощь с одним делом.",
]


def generate_from_templates(templates: List[str], n_samples: int = 1000) -> List[str]:
    """Генерация текстов из шаблонов с вариациями"""
    result = []
    n_per_template = n_samples // len(templates) + 1

    for template in templates:
        for _ in range(n_per_template):
            text = template
            if random.random() > 0.5:
                text = text.lower()
            if random.random() > 0.7:
                text = text.replace("!", ".")
            if random.random() > 0.8:
                text = add_noise(text, noise_prob=1.0)
            result.append(text)

    return result[:n_samples]


def prepare_dataset(
    output_path: str = "train.csv",
    n_samples: int = 3000,
    use_noise: bool = True,
    length_range: Tuple[int, int] = (8, 40),
) -> pd.DataFrame:
    """Полный pipeline подготовки данных"""
    print("=" * 60)
    print("🚀 ПОДГОТОВКА ДАННЫХ")
    print("=" * 60)

    formal_texts = load_formal_data(max_samples=n_samples * 2)
    informal_texts = load_informal_data(max_samples=n_samples * 2)

    MIN_SAMPLES = 500
    if len(formal_texts) < MIN_SAMPLES:
        print("\n⚠️ Добавляем formal шаблоны...")
        formal_texts.extend(generate_from_templates(FORMAL_TEMPLATES, MIN_SAMPLES))
    if len(informal_texts) < MIN_SAMPLES:
        print("\n⚠️ Добавляем informal шаблоны...")
        informal_texts.extend(generate_from_templates(INFORMAL_TEMPLATES, MIN_SAMPLES))

    formal_texts, informal_texts = normalize_lengths(
        formal_texts, informal_texts, target_range=length_range
    )

    if use_noise:
        print("\n🔊 Аугментация:")
        formal_texts = augment_with_noise(formal_texts, augment_factor=0.2)
        informal_texts = augment_with_noise(informal_texts, augment_factor=0.2)

    min_size = min(len(formal_texts), len(informal_texts))
    n_final = min(min_size, n_samples)

    formal_sample = random.sample(formal_texts, n_final)
    informal_sample = random.sample(informal_texts, n_final)

    formal_data = [{"text": t, "label": "formal"} for t in formal_sample]
    informal_data = [{"text": t, "label": "informal"} for t in informal_sample]

    df = pd.DataFrame(formal_data + informal_data)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего: {len(df)}")
    print(f"Классы: {df['label'].value_counts().to_dict()}")

    df["word_count"] = df["text"].str.split().str.len()
    print("\nДлина (слова):")
    print(df.groupby("label")["word_count"].agg(["mean", "std"]).round(1))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df[["text", "label"]].to_csv(output_path, index=False)
    print(f"\n✅ Сохранено: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="train.csv")
    parser.add_argument("--samples", "-n", type=int, default=3000)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=40)
    parser.add_argument("--no-noise", action="store_true")

    args = parser.parse_args()

    prepare_dataset(
        output_path=args.output,
        n_samples=args.samples,
        use_noise=not args.no_noise,
        length_range=(args.min_words, args.max_words),
    )


if __name__ == "__main__":
    main()
