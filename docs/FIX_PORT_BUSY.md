# Исправление ошибки "address already in use"

Если порт 8000 уже занят, выполните:

## 1. Найти процесс, занимающий порт 8000

```bash
# Вариант 1: Использование lsof
lsof -i :8000

# Вариант 2: Использование netstat
netstat -tulpn | grep :8000

# Вариант 3: Использование ss
ss -tulpn | grep :8000

# Вариант 4: Использование fuser
fuser 8000/tcp
```

## 2. Остановить процесс

```bash
# Если нашли PID (например, 12345), остановите его:
kill 12345

# Или принудительно:
kill -9 12345
```

## 3. Или изменить порт в .env

```bash
nano .env
# Измените:
PORT=8001
```

И обновите SERVER_URL в .env:
```
SERVER_URL=http://80.87.105.61:8001/predict
```

## 4. Проверка запущенных процессов Python

```bash
ps aux | grep python
ps aux | grep uvicorn
```

## 5. Остановка всех процессов uvicorn

```bash
pkill -f uvicorn
# или
killall uvicorn
```
