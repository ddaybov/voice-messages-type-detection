# Исправление проблемы "Empty text" (ASR не распознает речь)

## Проблема
Бот отвечает "Ошибка: Empty text" - ASR не может распознать текст из голосовых сообщений.

## Решения

### 1. Проверка настроек ASR в .env

```bash
nano .env
```

Убедитесь, что указано:
```
ASR_BACKEND=speech_recognition
ASR_LANGUAGE=ru-RU
```

### 2. Проблема с Google Speech Recognition

По умолчанию используется Google Speech Recognition, который:
- Требует интернет-соединение
- Может быть заблокирован или недоступен
- Может требовать API ключ

**Решение:** Установите PocketSphinx (офлайн распознавание):

```bash
source .venv/bin/activate
pip install --no-cache-dir pocketsphinx
```

### 3. Проверка работы ASR напрямую

Создайте тестовый скрипт:

```bash
cat > test_asr.py << 'EOF'
import os
os.environ['ASR_BACKEND'] = 'speech_recognition'
os.environ['ASR_LANGUAGE'] = 'ru-RU'

from server.audio_processor import audio_service
import tempfile

# Создайте тестовый WAV файл или используйте существующий
# test_audio.wav должен быть в формате WAV 16kHz mono

result = audio_service.transcribe('test_audio.wav', 'ru-RU')
if result:
    print(f"Text: {result.text}")
    print(f"Duration: {result.duration}")
    print(f"Words: {result.word_count}")
    print(f"Backend: {result.backend}")
else:
    print("ASR failed")
EOF

python test_asr.py
```

### 4. Альтернатива: Использование Whisper (более точное, но тяжелее)

```bash
source .venv/bin/activate
pip install --no-cache-dir openai-whisper
```

В .env:
```
ASR_BACKEND=whisper
WHISPER_MODEL=base  # или tiny для экономии места
```

### 5. Проверка логов сервера

```bash
# Если сервер запущен в screen
screen -r server

# Или проверьте логи systemd
journalctl -u voice-messages-server -f
```

### 6. Временное решение: Отключить проверку пустого текста

Если нужно протестировать бота без ASR, можно временно изменить код, но это не рекомендуется для продакшена.
