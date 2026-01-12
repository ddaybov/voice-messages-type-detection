# Решение конфликта git pull после рефакторинга

Если при выполнении `git pull` вы получили ошибку:
```
error: Your local changes to the following files would be overwritten by merge:
	install_pytorch_whisper.sh
```

## Решение

### Вариант 1: Удалить локальный файл (рекомендуется)

Если ваши локальные изменения в `install_pytorch_whisper.sh` не важны, просто удалите файл:

```bash
rm install_pytorch_whisper.sh
git pull
```

### Вариант 2: Сохранить изменения временно (stash)

Если вы хотите сохранить локальные изменения на будущее:

```bash
# Сохранить изменения
git stash

# Обновить код
git pull

# Посмотреть сохраненные изменения (опционально)
git stash show -p

# Если нужно восстановить изменения (опционально)
# git stash pop
```

### Вариант 3: Закоммитить локальные изменения

Если ваши изменения важны и должны быть сохранены:

```bash
# Посмотреть различия
git diff install_pytorch_whisper.sh

# Добавить файл в staging
git add install_pytorch_whisper.sh

# Закоммитить
git commit -m "Local changes to install_pytorch_whisper.sh"

# Теперь pull должен работать (может быть merge commit)
git pull

# Если будут конфликты в самом файле, разрешите их вручную
```

## После успешного git pull

После того как `git pull` выполнится успешно, убедитесь, что все изменения применены:

```bash
# Проверить статус
git status

# Выдать права на выполнение скриптам (если появилась ошибка "Permission denied")
chmod +x *.sh

# Перезапустить сервер и бота (если они запущены)
./stop_all.sh
./start_all.sh
```

### Если появилась ошибка "Permission denied"

При выполнении скриптов `./stop_all.sh` или `./start_all.sh` может появиться ошибка:
```
-bash: ./stop_all.sh: Permission denied
```

**Решение:** Выдайте права на выполнение:
```bash
chmod +x stop_all.sh start_all.sh
# или для всех скриптов сразу:
chmod +x *.sh
```

## Примечание

После рефакторинга структура проекта изменилась:
- Добавлены новые модули: `server/config.py`, `server/constants.py`, `server/utils.py`
- Добавлены модули для бота: `telegram/config.py`, `telegram/utils.py`
- Обновлены существующие файлы

Все настройки остаются в `.env` файле и продолжают работать как раньше.
