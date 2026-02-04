#!/bin/bash
# Скрипт для установки PyTorch (CPU) и Whisper (исправленная версия)

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Активация виртуального окружения
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ Виртуальное окружение не найдено. Запустите ./scripts/deploy.sh"
    exit 1
fi

echo "📦 Установка PyTorch (CPU версия) и Whisper..."
echo ""

# Проверка версии Python
PYTHON_VERSION=$(python --version 2>&1)
echo "Python версия: $PYTHON_VERSION"
echo ""

# Проверка свободного места
echo "📊 Проверка свободного места:"
df -h / | tail -1
echo ""

# 1. Установка PyTorch CPU версии
echo "1️⃣ Установка PyTorch (CPU версия)..."
echo "   Это может занять несколько минут..."

# Удаление старых nvidia пакетов если есть (они не нужны для CPU версии)
pip uninstall -y nvidia-cusparselt-cu12 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-nvtx-cu12 nvidia-nvshmem-cu12 nvidia-nvjitlink-cu12 nvidia-nccl-cu12 nvidia-cufile-cu12 2>/dev/null || true

# Попытка установки через основной PyPI (работает для всех версий Python)
echo "   Установка через основной PyPI..."
if pip install --no-cache-dir torch torchvision torchaudio 2>&1 | tee /tmp/pytorch_install.log; then
    echo "✅ PyTorch установлен через основной PyPI"
else
    echo "   Попытка установки только torch..."
    pip install --no-cache-dir torch || {
        echo "❌ Ошибка установки PyTorch. Проверьте логи выше."
        exit 1
    }
    echo "✅ PyTorch установлен"
fi
echo ""

# 2. Установка Whisper
echo "2️⃣ Установка Whisper..."
pip install --no-cache-dir openai-whisper || {
    echo "❌ Ошибка установки Whisper"
    exit 1
}
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
    if grep -q "^ASR_BACKEND" .env; then
        # Заменяем существующую строку
        sed -i 's/^ASR_BACKEND=.*/ASR_BACKEND=whisper/' .env
    else
        # Добавляем новую строку
        echo "ASR_BACKEND=whisper" >> .env
    fi
    
    # Добавляем WHISPER_MODEL если нет
    if ! grep -q "^WHISPER_MODEL" .env; then
        echo "WHISPER_MODEL=tiny" >> .env
    fi
    
    echo "✅ .env обновлен"
    echo ""
    echo "Текущие настройки ASR:"
    grep "^ASR_BACKEND\|^WHISPER_MODEL" .env || true
else
    echo "⚠️  .env файл не найден, создайте его вручную"
fi

echo ""
echo "✅ Установка завершена!"
echo ""
echo "💡 Следующие шаги:"
echo "1. Перезапустите сервер и бота: ./scripts/stop_all.sh && ./scripts/start_all.sh"
echo "2. Попробуйте отправить голосовое сообщение боту"
echo ""
echo "📊 Использование диска после установки:"
df -h / | tail -1
