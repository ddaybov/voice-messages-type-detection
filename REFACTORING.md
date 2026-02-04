# Рефакторинг проекта

## Основные изменения

### 0. Последний рефакторинг (структура и сервер)

- **Схемы API** — Pydantic-модели ответов вынесены в `server/schemas.py` (Health, PredictResponse, PredictTextResponse). В `main.py` используются только импорты из schemas.
- **ModelFactory и конфиг** — при инициализации сервера вызывается `get_factory(config.model.models_dir)`, путь к моделям задаётся через `server/config.py` (переменная `MODELS_DIR`).
- **Устаревший код** — `server/classifier.py` помечен как DEPRECATED: не используется (классификация идёт через `ml.model_factory` и пакет `ml`). Файл оставлен для справки, при желании можно удалить.
- **Документация** — вся fix/deploy и прочая документация перенесена в каталог `docs/`: FIX_*.md, CHECK_*.md, ASR_DIFFERENCES.md, DEPLOY.md, QUICK_START.md, START_AFTER_INSTALL.md, CURSOR_DATA_PREPARATION.md, FREE_SPACE.md.
- **Скрипты** — скрипты запуска и развёртывания перенесены в `scripts/`: start_server.sh, start_bot.sh, start_all.sh, stop_all.sh, deploy.sh, QUICK_FIX_SERVER_URL.sh, cleanup_disk.sh, install_pytorch_whisper*.sh. В каждом скрипте используется переход в корень проекта `ROOT="$(cd "$(dirname "$0")/.." && pwd)"` и `cd "$ROOT"`. Запуск: `./scripts/deploy.sh`, `./scripts/start_server.sh` и т.д. Описание — в `scripts/README.md`.

### 1. Централизованная конфигурация (`server/config.py`)

Создан модуль `config.py` с классами конфигурации:
- `ServerConfig` - настройки сервера
- `ModelConfig` - настройки моделей
- `ASRConfig` - настройки ASR
- `FFmpegConfig` - настройки FFmpeg
- `AppConfig` - главная конфигурация

Все настройки теперь загружаются из переменных окружения через `AppConfig.from_env()`.

### 2. Константы (`server/constants.py`)

Вынесены все константы в отдельный модуль:
- `SUPPORTED_AUDIO_FORMATS` - поддерживаемые форматы аудио
- `SUPPORTED_LANGUAGES` - поддерживаемые языки
- `MODEL_ALIASES` - алиасы моделей
- Константы по умолчанию

### 3. Утилиты (`server/utils.py`, `telegram/utils.py`)

Созданы модули с утилитарными функциями:
- `cleanup_files()` - безопасная очистка файлов
- `download_file()` - загрузка файлов из Telegram
- `truncate_message()` - обрезка сообщений для Telegram

### 4. Улучшения кода

#### `server/main.py`:
- Улучшена обработка ошибок с использованием try/except/finally
- Добавлена документация функций
- Использование централизованной конфигурации
- Улучшено логирование

#### `server/classifier.py`:
- Убрано дублирование ключа "ensemble" в словаре
- Использование централизованной конфигурации
- Улучшена обработка ошибок
- Добавлена документация методов

#### `server/audio_processor.py`:
- Использование централизованной конфигурации
- Улучшено логирование для всех ASR бэкендов
- Добавлена обработка ошибок для Whisper

#### `telegram/bot.py`:
- Вынесены константы в `telegram/config.py`
- Создан модуль `telegram/utils.py` для утилит
- Улучшена структура кода
- Использование констант из `server/constants.py`

### 5. Структура проекта

```
voice-messages-type-detection/
├── docs/                  # Документация (FIX_*, DEPLOY, QUICK_START и т.д.)
├── scripts/               # Скрипты запуска и развёртывания (см. scripts/README.md)
├── server/
│   ├── __init__.py
│   ├── config.py          # Конфигурация
│   ├── constants.py       # Константы
│   ├── schemas.py         # Pydantic-схемы ответов API
│   ├── utils.py           # Утилиты
│   ├── main.py            # FastAPI приложение
│   ├── audio_processor.py # Обработка аудио и ASR
│   └── classifier.py      # DEPRECATED — не используется
├── bot/
│   ├── config.py          # Конфигурация бота
│   ├── bot.py             # Telegram бот
│   └── check_server.py    # Проверка сервера
└── ...
```

## Преимущества рефакторинга

1. **Централизованная конфигурация** - все настройки в одном месте
2. **Улучшенная типизация** - использование dataclasses и type hints
3. **Лучшая организация кода** - разделение на модули
4. **Переиспользование кода** - утилиты вынесены в отдельные модули
5. **Улучшенная обработка ошибок** - более понятные сообщения
6. **Легче тестировать** - модульная структура
7. **Легче расширять** - понятная архитектура

## Миграция

Код обратно совместим. Все существующие `.env` файлы продолжат работать, так как конфигурация загружается из тех же переменных окружения.
