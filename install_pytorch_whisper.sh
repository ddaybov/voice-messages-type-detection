#!/bin/bash
# Скрипт для установки PyTorch (CPU) и Whisper

set -e

cd "$(dirname "$0")"

# Активация виртуального окружения
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ Виртуальное окружение не найдено. Запустите ./deploy.sh"
    exit 1
fi

echo "📦 Установка PyTorch (CPU версия) и Whisper..."
echo ""

# Проверка свободного места
echo "📊 Проверка свободного места:"
df -h / | tail -1
echo ""

# 1. Установка PyTorch CPU версии (без CUDA, намного меньше)
echo "1️⃣ Установка PyTorch (CPU версия)..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
echo "✅ PyTorch установлен"
echo ""

# 2. Установка Whisper
echo "2️⃣ Установка Whisper..."
pip install --no-cache-dir openai-whisper
echo "✅ Whisper установлен"
echo ""

# 3. Проверка установки
echo "3️⃣ Проверка установки..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} установлен')" || echo "❌ PyTorch не установлен"
python -c "import whisper; print(f'✅ Whisper установлен')" || echo "❌ Whisper не установлен"
echo ""

# 4. Обновление .env для использования Whisper
echo "4️⃣ Обновление .env для использования Whisper..."
if [ -f ".env" ]; then
    # Проверяем, есть ли уже ASR_BACKEND
    if grep -q "ASR_BACKEND" .env; then
        # Заменяем существующую строку
        sed -i 's/^ASR_BACKEND=.*/ASR_BACKEND=whisper/' .env
    else
        # Добавляем новую строку
        echo "ASR_BACKEND=whisper" >> .env
    fi
    
    # Добавляем WHISPER_MODEL если нет
    if ! grep -q "WHISPER_MODEL" .env; then
        echo "WHISPER_MODEL=tiny" >> .env
    fi
    
    echo "✅ .env обновлен"
    echo ""
    echo "Текущие настройки ASR:"
    grep "ASR_BACKEND\|WHISPER_MODEL" .env || true
else
    echo "⚠️  .env файл не найден, создайте его вручную"
fi

echo ""
echo "✅ Установка завершена!"
echo ""
echo "💡 Следующие шаги:"
echo "1. Перезапустите сервер и бота: ./stop_all.sh && ./start_all.sh"
echo "2. Попробуйте отправить голосовое сообщение боту"
echo ""
echo "📊 Использование диска после установки:"
df -h / | tail -1
