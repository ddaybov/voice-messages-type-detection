# Освобождение места на диске

Если возникает ошибка "No space left on device", выполните:

## 1. Проверка использования диска

```bash
df -h
du -sh /root/* | sort -h
```

## 2. Очистка кэша pip

```bash
pip cache purge
# или
rm -rf ~/.cache/pip
```

## 3. Очистка временных файлов

```bash
# Очистка системных временных файлов
sudo apt-get clean
sudo apt-get autoclean
sudo apt-get autoremove

# Очистка временных файлов
rm -rf /tmp/*
rm -rf /var/tmp/*
```

## 4. Очистка логов (осторожно!)

```bash
# Просмотр размера логов
du -sh /var/log/*

# Очистка старых логов (оставит только последние)
journalctl --vacuum-time=3d
```

## 5. Установка только критичных зависимостей

Вместо всех зависимостей, установите только необходимое:

```bash
source .venv/bin/activate
pip install --no-cache-dir uvicorn[standard] fastapi pydantic
pip install --no-cache-dir python-telegram-bot aiohttp
pip install --no-cache-dir pydub SpeechRecognition
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir transformers numpy scikit-learn joblib
```

Флаг `--no-cache-dir` предотвратит сохранение кэша pip.
