# Инструкция по развертыванию на сервере

## Требования

- Ubuntu (или другой Linux дистрибутив)
- Python 3.8+
- pip
- Доступ к серверу по SSH

## Быстрое развертывание

### 1. Подключение к серверу

```bash
ssh root@80.87.105.61
```

### 2. Клонирование репозитория

```bash
cd /root
git clone https://github.com/ddaybov/voice-messages-type-detection.git
cd voice-messages-type-detection
```

### 3. Установка зависимостей

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### 4. Настройка .env файла

```bash
nano .env
```

Убедитесь, что указаны:
- `BOT_TOKEN` - токен бота от @BotFather
- `SERVER_URL` - URL сервера (например, `http://80.87.105.61:8000/predict`)
- `ASR_BACKEND` - бэкенд распознавания речи
- `DEFAULT_MODEL` - модель по умолчанию

### 5. Запуск через systemd (рекомендуется)

```bash
# Копирование сервисов
sudo cp systemd/*.service /etc/systemd/system/

# Перезагрузка systemd
sudo systemctl daemon-reload

# Запуск сервисов
sudo systemctl start voice-messages-server
sudo systemctl start voice-messages-bot

# Включение автозапуска
sudo systemctl enable voice-messages-server
sudo systemctl enable voice-messages-bot

# Проверка статуса
sudo systemctl status voice-messages-server
sudo systemctl status voice-messages-bot
```

### 6. Просмотр логов

```bash
# Логи сервера
sudo journalctl -u voice-messages-server -f

# Логи бота
sudo journalctl -u voice-messages-bot -f
```

## Альтернативный запуск (без systemd)

### Запуск в screen/tmux

```bash
# Установка screen (если не установлен)
sudo apt-get install screen

# Запуск сервера в screen
screen -S server
./start_server.sh
# Нажмите Ctrl+A, затем D для отсоединения

# Запуск бота в screen
screen -S bot
./scripts/start_bot.sh
# Нажмите Ctrl+A, затем D для отсоединения

# Подключение к сессиям
screen -r server
screen -r bot
```

### Запуск в фоне (nohup)

```bash
# Запуск сервера
nohup ./start_server.sh > server.log 2>&1 &

# Запуск бота
nohup ./scripts/start_bot.sh > bot.log 2>&1 &
```

## Проверка работы

### Проверка сервера

```bash
# Локально на сервере
curl http://localhost:8000/health

# Извне (если порт открыт)
curl http://80.87.105.61:8000/health
```

### Проверка бота

Используйте команду `/status` в Telegram боте.

Или запустите скрипт проверки:

```bash
python telegram/check_server.py
```

## Обновление

```bash
cd /root/voice-messages-type-detection
git pull
source .venv/bin/activate
pip install -r requirements.txt

# Перезапуск сервисов
sudo systemctl restart voice-messages-server
sudo systemctl restart voice-messages-bot
```

## Устранение проблем

### Сервер не запускается

1. Проверьте логи:
   ```bash
   sudo journalctl -u voice-messages-server -n 50
   ```

2. Проверьте порт:
   ```bash
   netstat -tulpn | grep 8000
   ```

3. Проверьте .env файл:
   ```bash
   cat .env
   ```

### Бот не отвечает

1. Проверьте логи:
   ```bash
   sudo journalctl -u voice-messages-bot -n 50
   ```

2. Проверьте BOT_TOKEN:
   ```bash
   grep BOT_TOKEN .env
   ```

3. Проверьте доступность сервера:
   ```bash
   python telegram/check_server.py
   ```

### Проблемы с зависимостями

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Безопасность

1. **Firewall**: Убедитесь, что порт 8000 открыт только для необходимых IP
   ```bash
   sudo ufw allow 8000/tcp
   ```

2. **.env файл**: Не коммитьте .env файл в git
   ```bash
   echo ".env" >> .gitignore
   ```

3. **Права доступа**: Ограничьте доступ к .env
   ```bash
   chmod 600 .env
   ```

## Мониторинг

### Проверка использования ресурсов

```bash
# CPU и память
top

# Дисковое пространство
df -h

# Процессы Python
ps aux | grep python
```

### Автоматический перезапуск при сбое

Systemd сервисы автоматически перезапускаются при сбое (благодаря `Restart=always`).
