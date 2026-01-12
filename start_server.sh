#!/bin/bash
# Скрипт для запуска FastAPI сервера

set -e

# Переход в директорию скрипта
cd "$(dirname "$0")"

# Активация виртуального окружения
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ Виртуальное окружение не найдено. Запустите ./deploy.sh"
    exit 1
fi

# Загрузка переменных окружения
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  .env файл не найден"
fi

# Параметры запуска
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
LOG_LEVEL=${LOG_LEVEL:-"info"}

echo "🚀 Запуск FastAPI сервера..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Log Level: $LOG_LEVEL"
echo ""

# Запуск сервера
uvicorn server.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL"
