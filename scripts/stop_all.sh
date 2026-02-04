#!/bin/bash
# Скрипт для остановки сервера и бота

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "🛑 Остановка сервера и бота..."

# Остановка по PID файлам
if [ -f "server.pid" ]; then
    SERVER_PID=$(cat server.pid)
    if kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID
        echo "✅ Сервер остановлен (PID: $SERVER_PID)"
    else
        echo "⚠️  Сервер уже не запущен"
    fi
    rm -f server.pid
else
    echo "⚠️  server.pid не найден, пытаюсь найти процесс..."
    pkill -f "uvicorn server.main:app" && echo "✅ Сервер остановлен" || echo "⚠️  Процесс сервера не найден"
fi

if [ -f "bot.pid" ]; then
    BOT_PID=$(cat bot.pid)
    if kill -0 $BOT_PID 2>/dev/null; then
        kill $BOT_PID
        echo "✅ Бот остановлен (PID: $BOT_PID)"
    else
        echo "⚠️  Бот уже не запущен"
    fi
    rm -f bot.pid
else
    echo "⚠️  bot.pid не найден, пытаюсь найти процесс..."
    pkill -f "bot.bot" && echo "✅ Бот остановлен" || echo "⚠️  Процесс бота не найден"
fi

echo ""
echo "✅ Готово!"
