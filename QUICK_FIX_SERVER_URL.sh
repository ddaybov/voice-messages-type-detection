#!/bin/bash
# Быстрое исправление SERVER_URL в .env файле

cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
    echo "❌ Файл .env не найден!"
    exit 1
fi

echo "🔧 Обновление SERVER_URL в .env..."

# Читаем текущий PORT из .env
PORT=$(grep "^PORT=" .env | cut -d'=' -f2 | tr -d '"' || echo "8001")
if [ -z "$PORT" ]; then
    PORT="8001"
fi

# Определяем, использовать ли внешний IP или localhost
# Проверяем, есть ли уже SERVER_URL с IP
if grep -q "SERVER_URL=http://80.87.105.61" .env; then
    echo "✅ SERVER_URL уже содержит IP 80.87.105.61"
    # Обновляем только порт если нужно
    sed -i "s|SERVER_URL=http://80.87.105.61:[0-9]*|SERVER_URL=http://80.87.105.61:${PORT}|" .env
elif grep -q "SERVER_URL=http://127.0.0.1" .env; then
    echo "ℹ️  Текущий SERVER_URL использует localhost (127.0.0.1)"
    echo "   Хотите изменить на внешний IP 80.87.105.61? (y/n)"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        sed -i "s|SERVER_URL=http://127.0.0.1:[0-9]*|SERVER_URL=http://80.87.105.61:${PORT}|" .env
        echo "✅ SERVER_URL обновлен на http://80.87.105.61:${PORT}"
    else
        sed -i "s|SERVER_URL=http://127.0.0.1:[0-9]*|SERVER_URL=http://127.0.0.1:${PORT}|" .env
        echo "✅ Порт в SERVER_URL обновлен на ${PORT}"
    fi
elif grep -q "^SERVER_URL=" .env; then
    # SERVER_URL существует, обновляем порт
    sed -i "s|^SERVER_URL=.*|SERVER_URL=http://80.87.105.61:${PORT}|" .env
    echo "✅ SERVER_URL обновлен на http://80.87.105.61:${PORT}"
else
    # SERVER_URL не существует, добавляем
    echo "SERVER_URL=http://80.87.105.61:${PORT}" >> .env
    echo "✅ SERVER_URL добавлен: http://80.87.105.61:${PORT}"
fi

# Убедимся, что PORT установлен
if ! grep -q "^PORT=" .env; then
    echo "PORT=${PORT}" >> .env
    echo "✅ PORT добавлен: ${PORT}"
fi

echo ""
echo "📋 Текущая конфигурация:"
grep -E "^(SERVER_URL|PORT)=" .env

echo ""
echo "✅ Готово! Перезапустите сервисы:"
echo "   ./stop_all.sh"
echo "   ./start_all.sh"
