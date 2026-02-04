# Скрипты

Скрипты запуска, развёртывания и обслуживания. Запускать из корня проекта или по полному пути.

| Скрипт | Назначение |
|--------|------------|
| `deploy.sh` | Развёртывание: venv, зависимости, .env |
| `start_server.sh` | Запуск FastAPI-сервера |
| `start_bot.sh` | Запуск Telegram-бота |
| `start_all.sh` | Запуск сервера и бота в фоне |
| `stop_all.sh` | Остановка сервера и бота |
| `QUICK_FIX_SERVER_URL.sh` | Обновление SERVER_URL в .env |
| `cleanup_disk.sh` | Очистка диска перед установкой PyTorch/Whisper |
| `install_pytorch_whisper.sh` | Установка PyTorch (CPU) и Whisper |
| `install_pytorch_whisper_fixed.sh` | Установка PyTorch/Whisper (альтернативный вариант) |

Примеры (из корня проекта):

```bash
./scripts/deploy.sh
./scripts/start_server.sh
./scripts/start_bot.sh
./scripts/start_all.sh
./scripts/stop_all.sh
```
