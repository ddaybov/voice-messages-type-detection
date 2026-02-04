# Исправление ошибки ImportError в боте

## Проблема

После рефакторинга появилась ошибка:
```
ImportError: cannot import name 'Update' from 'telegram' 
(/root/voice-messages-type-detection/telegram/__init__.py)
```

## Причина

Локальная директория `telegram/` конфликтует с установленной библиотекой `python-telegram-bot` (которая также называется `telegram`). Когда бот запускается как скрипт (`python telegram/bot.py`), Python пытается импортировать из локальной директории, а не из установленной библиотеки.

## Решение

Бот теперь запускается как модуль Python: `python -m telegram.bot` вместо `python telegram/bot.py`.

## Что нужно сделать на сервере

1. **Обновите код:**
   ```bash
   git pull
   ```

2. **Перезапустите бота:**
   ```bash
   ./stop_all.sh
   ./start_all.sh
   ```

3. **Проверьте логи:**
   ```bash
   tail -f bot.log
   ```

4. **Убедитесь, что бот работает:**
   - Отправьте команду `/start` боту в Telegram
   - Проверьте логи на наличие ошибок

## Изменения в коде

- `start_bot.sh`: изменен способ запуска на `python -m telegram.bot`
- `start_all.sh`: изменен способ запуска на `python -m telegram.bot`
- `systemd/voice-messages-bot.service`: обновлен ExecStart
- `stop_all.sh`: обновлен pkill паттерн
- `telegram/bot.py`: упрощены импорты (теперь только относительные)

## Примечание

Запуск как модуль (`python -m telegram.bot`) - это правильный способ для пакетов Python, который избегает конфликтов имен и правильно обрабатывает относительные импорты.
