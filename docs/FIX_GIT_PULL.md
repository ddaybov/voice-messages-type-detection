# Решение проблемы git pull

Если при `git pull` возникает ошибка:
```
error: The following untracked working tree files would be overwritten by merge:
	deploy.sh
	start_bot.sh
	start_server.sh
```

## Решение на сервере:

Выполните на сервере одну из команд:

### Вариант 1: Удалить локальные файлы (рекомендуется)
```bash
cd /root/voice-messages-type-detection
rm deploy.sh start_bot.sh start_server.sh
git pull
```

### Вариант 2: Переместить файлы в резервную копию
```bash
cd /root/voice-messages-type-detection
mkdir -p backup
mv deploy.sh start_bot.sh start_server.sh backup/
git pull
```

### Вариант 3: Принудительно перезаписать (если уверены)
```bash
cd /root/voice-messages-type-detection
git fetch origin
git reset --hard origin/main
```

После этого все файлы будут синхронизированы с репозиторием.
