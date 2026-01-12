# Быстрый старт на сервере

## Шаг 1: Подключение к серверу

```bash
ssh root@80.87.105.61
```

## Шаг 2: Клонирование и установка

```bash
cd /root
git clone https://github.com/ddaybov/voice-messages-type-detection.git
cd voice-messages-type-detection
chmod +x deploy.sh start_server.sh start_bot.sh
./deploy.sh
```

## Шаг 3: Настройка .env

```bash
nano .env
```

**Обязательно укажите:**
- `BOT_TOKEN=ваш_токен_от_BotFather`
- `SERVER_URL=http://80.87.105.61:8000/predict` (или `http://127.0.0.1:8000/predict` если только локально)

## Шаг 4: Установка systemd сервисов

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable voice-messages-server voice-messages-bot
sudo systemctl start voice-messages-server
sudo systemctl start voice-messages-bot
```

## Шаг 5: Проверка

```bash
# Проверка статуса
sudo systemctl status voice-messages-server
sudo systemctl status voice-messages-bot

# Проверка логов
sudo journalctl -u voice-messages-server -f
sudo journalctl -u voice-messages-bot -f

# Проверка сервера
curl http://localhost:8000/health
```

## Полезные команды

```bash
# Остановка
sudo systemctl stop voice-messages-server
sudo systemctl stop voice-messages-bot

# Перезапуск
sudo systemctl restart voice-messages-server
sudo systemctl restart voice-messages-bot

# Просмотр логов
sudo journalctl -u voice-messages-server -n 100
sudo journalctl -u voice-messages-bot -n 100
```

## Альтернатива: Запуск без systemd

Если не хотите использовать systemd, можно запустить в screen:

```bash
# Установка screen
sudo apt-get install screen

# Запуск сервера
screen -S server -d -m bash -c "cd /root/voice-messages-type-detection && source .venv/bin/activate && ./start_server.sh"

# Запуск бота
screen -S bot -d -m bash -c "cd /root/voice-messages-type-detection && source .venv/bin/activate && ./start_bot.sh"

# Просмотр сессий
screen -ls

# Подключение к сессии
screen -r server
screen -r bot
```
