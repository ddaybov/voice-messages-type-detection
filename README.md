# Voice Messages Type Detection

Классификация типа голосовых сообщений (формальный / неформальный стиль) с конвертацией аудио в текст и интеграцией с Telegram-ботом.

## Структура
```
telegram/        # телеграм-бот
server/          # FastAPI: /health, /supported_formats, /predict (включая ансамбль)
ml/              # обучение: CNN, трансформеры
models/          # артефакты моделей и ensemble.json
data/            # датасеты
```
Диаграмма архитектуры намеренно не включена.

## Запуск
1) Установка:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
2) Обучение авторской модели (пример):
```bash
python ml/train_daybov_model.py --data data/train.csv --outdir models/
```
3) Сервер:
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```
4) Телеграм-бот:
```bash
python telegram/bot.py
```

## Обучение и метрики
Скрипты в `ml/`:
- `train_daybov_model.py` — Daybov TF‑IDF+LR.
- `train_classic.py` — логистическая регрессия, SVM, Naive Bayes.
- `train_cnn.py` — CNN (PyTorch).
- `train_transformer.py` — трансформеры (HF).
- `evaluate.py` — оценка sklearn‑моделей.

Метрики (test): macro‑F1, accuracy, weighted‑F1, ROC‑AUC (для бинарной), PR‑AUC, confusion matrix.

## ASR
Выбор бэкенда в `.env` через `ASR_BACKEND`: `speech_recognition`, `vosk`, `whisper`.
Конвертация в WAV 16 kHz mono выполняется автоматически (FFmpeg).

## Telegram-бот
Принимает голосовые/аудиофайлы, отправляет на `/predict`, поддерживает выбор модели и ансамбля. Возвращает класс, уверенность, длительность, число слов и превью текста.

## Ансамбль
Конфигурация `models/ensemble.json`:
```json
{
  "models": ["daybov", "logreg", "cnn", "bert"],
  "weights": {"daybov": 1.2, "logreg": 1.0, "cnn": 0.8, "bert": 1.3}
}
```

## Формат датасета
CSV:
```csv
text,label
"Здравствуйте, подскажите статус...",formal
"Привет! Как дела?",informal
```

## Публикация в GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/DimaDaybov/voice-messages-type-detection.git
git push -u origin main
```

## Лицензия
MIT.
