#!/bin/bash
# Скрипт для развертывания бота на сервере

set -e

echo "🚀 Начало развертывания..."

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 не установлен${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Найден: ${PYTHON_VERSION}${NC}"

# Проверка pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}⚠️  pip3 не найден, устанавливаю...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Создание виртуального окружения
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}📦 Создаю виртуальное окружение...${NC}"
    python3 -m venv .venv
fi

# Активация виртуального окружения
echo -e "${YELLOW}🔌 Активация виртуального окружения...${NC}"
source .venv/bin/activate

# Обновление pip
echo -e "${YELLOW}⬆️  Обновляю pip...${NC}"
pip install --upgrade pip

# Установка зависимостей
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}📥 Устанавливаю зависимости...${NC}"
    pip install -r requirements.txt
else
    echo -e "${RED}❌ requirements.txt не найден${NC}"
    exit 1
fi

# Проверка .env файла
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}📝 Создаю .env из .env.example...${NC}"
        cp .env.example .env
        echo -e "${RED}⚠️  ВАЖНО: Отредактируйте .env файл и укажите BOT_TOKEN и другие настройки!${NC}"
    else
        echo -e "${RED}❌ .env.example не найден${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Развертывание завершено!${NC}"
echo ""
echo "Следующие шаги:"
echo "1. Отредактируйте .env файл: nano .env"
echo "2. Запустите сервер: ./start_server.sh"
echo "3. Запустите бота: ./start_bot.sh"
echo ""
echo "Или используйте systemd сервисы:"
echo "  sudo systemctl start voice-messages-server"
echo "  sudo systemctl start voice-messages-bot"
