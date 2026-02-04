# Проверка и исправление проблемы распознавания речи (ASR)

## Проблема

Бот работает, но не распознает текст из голосовых сообщений. Ошибка: "Не удалось распознать текст из аудио".

## Диагностика

### 1. Проверьте логи сервера

```bash
tail -50 server.log
```

Ищите:
- Сообщения об ошибках ASR
- Какой бэкенд используется
- Детали ошибок распознавания

### 2. Проверьте настройки ASR в .env

```bash
cat .env | grep ASR
```

Должны быть установлены:
- `ASR_BACKEND` - какой бэкенд использовать (speech_recognition, whisper, vosk)
- `ASR_LANGUAGE` - язык (например, ru-RU)
- `WHISPER_MODEL` - если используется whisper (tiny, base, small, medium, large)

### 3. Проверьте доступность ASR бэкенда

**Для speech_recognition:**
```bash
python -c "import speech_recognition; print('OK')"
```

**Для whisper:**
```bash
python -c "import whisper; print('OK')"
```

### 4. Проверьте формат аудио

```bash
# Проверьте, что FFmpeg установлен
ffmpeg -version

# Проверьте, что pydub работает
python -c "from pydub import AudioSegment; print('OK')"
```

## Решение

### Вариант 1: Использовать Whisper (рекомендуется)

1. **Убедитесь, что Whisper установлен:**
   ```bash
   pip install openai-whisper
   ```

2. **Обновите .env:**
   ```bash
   ASR_BACKEND=whisper
   ASR_LANGUAGE=ru-RU
   WHISPER_MODEL=tiny
   ```

3. **Перезапустите сервер:**
   ```bash
   ./stop_all.sh
   ./start_all.sh
   ```

### Вариант 2: Использовать speech_recognition (Google)

1. **Проверьте .env:**
   ```bash
   ASR_BACKEND=speech_recognition
   ASR_LANGUAGE=ru-RU
   ```

2. **Перезапустите сервер:**
   ```bash
   ./stop_all.sh
   ./start_all.sh
   ```

### Вариант 3: Использовать Vosk (офлайн)

1. **Установите Vosk:**
   ```bash
   pip install vosk
   ```

2. **Скачайте модель для русского языка:**
   ```bash
   # Создайте директорию для моделей
   mkdir -p vosk-models
   cd vosk-models
   
   # Скачайте модель (пример URL, проверьте актуальный)
   wget https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip
   unzip vosk-model-ru-0.22.zip
   ```

3. **Обновите .env:**
   ```bash
   ASR_BACKEND=vosk
   ASR_LANGUAGE=ru-RU
   VOSK_MODEL_PATH=/root/voice-messages-type-detection/vosk-models/vosk-model-ru-0.22
   ```

4. **Перезапустите сервер:**
   ```bash
   ./stop_all.sh
   ./start_all.sh
   ```

## Проверка работы

После настройки:

1. **Отправьте голосовое сообщение боту**

2. **Проверьте логи:**
   ```bash
   tail -f server.log
   ```

3. **Ищите в логах:**
   - "Transcribing audio: ..."
   - "ASR result: success=True"
   - "Whisper transcription completed" (если используется Whisper)
   - Детали ошибок, если они есть

## Типичные проблемы

### 1. Whisper не установлен
```bash
# Решение:
pip install openai-whisper
```

### 2. Speech Recognition не работает (нет интернета или проблемы с Google API)
```bash
# Решение: используйте Whisper или Vosk
```

### 3. FFmpeg не установлен
```bash
# Решение:
apt-get update && apt-get install -y ffmpeg
```

### 4. Аудио слишком короткое или тихое
- Убедитесь, что голосовое сообщение достаточно длинное (минимум 1-2 секунды)
- Говорите четко и достаточно громко
