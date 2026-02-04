# Исправление прав доступа к скриптам

Если возникает ошибка `Permission denied` при запуске скриптов, выполните:

```bash
cd /root/voice-messages-type-detection
chmod +x deploy.sh start_server.sh start_bot.sh
```

Или одной командой для всех .sh файлов:

```bash
chmod +x *.sh
```

После этого скрипты можно будет запускать:

```bash
./start_server.sh
./start_bot.sh
```
