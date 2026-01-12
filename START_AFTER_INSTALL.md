# Запуск после установки зависимостей

## 1. Проверка установки

```bash
cd /root/voice-messages-type-detection
source .venv/bin/activate

# Проверка основных команд
uvicorn --version
python -c "import fastapi; print('FastAPI OK')"
python -c "import telegram; print('Telegram bot OK')"
python -c "import sklearn; print('sklearn OK')"
```

## 2. Настройка .env

```bash
nano .env
```

Убедитесь, что указаны:
- `BOT_TOKEN=ваш_токен`
- `SERVER_URL=http://80.87.105.61:8000/predict` (или `http://127.0.0.1:8000/predict`)
- `DEFAULT_MODEL=logreg` (так как PyTorch не установлен, используйте sklearn модели)

## 3. Запуск сервера

```bash
./start_server.sh
```

Или в screen:
```bash
screen -S server -d -m bash -c "cd /root/voice-messages-type-detection && source .venv/bin/activate && ./start_server.sh"
```

## 4. Запуск бота

В другом терминале или screen:
```bash
./start_bot.sh
```

Или в screen:
```bash
screen -S bot -d -m bash -c "cd /root/voice-messages-type-detection && source .venv/bin/activate && ./start_bot.sh"
```

## 5. Проверка работы

```bash
# Проверка сервера
curl http://localhost:8000/health

# Проверка через бота
# Отправьте /status в Telegram боту
```

## Доступные модели

С установленными зависимостями работают:
- ✅ `logreg` - Logistic Regression
- ✅ `svm` - SVM
- ✅ `nb` - Naive Bayes
- ✅ `ensemble` - если в ensemble.json только sklearn модели

Не работают (требуют PyTorch):
- ❌ `daybovnet` - требует PyTorch
- ❌ `dimanet` - требует PyTorch
- ❌ `bert` - требует transformers (PyTorch)
