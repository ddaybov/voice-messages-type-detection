# Отличия настроек распознавания аудио от «первого бота»

## Как было при первом боте (рабочий вариант)

| Параметр | Значение |
|----------|----------|
| **ASR_BACKEND** | `speech_recognition` (по умолчанию в config) |
| **Источник** | Один бот → сервер → Google Speech Recognition |
| **Конвертация** | OGG → WAV через pydub (16 kHz, моно); про `set_sample_width(2)` в старом коде могло не быть |
| **Язык** | `ru-RU` через `recognize_google(audio, language=lang)` |
| **Fallback** | При ошибке Google — PocketSphinx (для русского нет данных, поэтому часто пусто) |

## Как сейчас (после доработок и второго бота)

| Параметр | Значение |
|----------|----------|
| **ASR_BACKEND** | Задаётся в .env или при запуске (`whisper` / `speech_recognition`) |
| **Источник** | Два бота (bot + bot_v3), оба шлют тот же формат: `filename="audio.ogg"`, поле `file` |
| **Конвертация** | OGG → WAV: `set_frame_rate(16000).set_channels(1).set_sample_width(2)` (16-bit) |
| **Whisper** | Модель из `WHISPER_MODEL` (tiny/base/small); по умолчанию в config — `tiny` |
| **Для Whisper** | Сейчас передаётся **исходный файл (OGG)** в Whisper, чтобы он сам грузил через свой ffmpeg (без нашей конвертации в WAV) — так проверяем, не ломает ли конвертация pydub |

## Что могло измениться и ломать распознавание

1. **Backend**  
   Раньше по умолчанию использовался **Google** (`speech_recognition`). Сейчас часто запускают с **Whisper** (`ASR_BACKEND=whisper`). Whisper у тебя даёт пустой текст (мы уже пробовали `no_speech_threshold=0`, `initial_prompt`, `language=None`).

2. **Конвертация OGG → WAV**  
   В логах видно вызов ffmpeg с `pcm_s32le` при загрузке OGG в pydub. Потом мы делаем `set_sample_width(2)` и экспортируем WAV. Теоретически это могло отличаться от того, как было при первом боте (если тогда не было 16-bit).  
   **Сделано:** для Whisper теперь можно подавать **исходный OGG** (`src_path`), чтобы Whisper сам декодировал через свой `load_audio` (ffmpeg). Если с OGG текст появится — проблема была в нашей конвертации в WAV.

3. **Модель Whisper**  
   В `config.py` по умолчанию `whisper_model: str = "tiny"`. Для русского в FIX_WHISPER_EMPTY_TEXT.md рекомендуют `base` или `small`. На сервере у тебя уже `small`. Модель не кэшировалась — сейчас кэш есть (`_whisper_model`).

4. **Боты и формат запроса**  
   Оба бота шлют одно и то же: `FormData`, поле `file` (байты), `filename="audio.ogg"`. Сервер сохраняет во временный файл с суффиксом из имени (`.ogg`) и раньше всегда конвертировал в WAV. Логика приёма не менялась.

5. **Минимальная длительность/слова**  
   `MIN_DURATION`, `MIN_WORDS` по умолчанию 0 — на ответ «пустой текст» не влияют.

## Что сделано в коде для сравнения с «первым ботом»

- **Whisper:** при наличии `src_path` вызывается `_transcribe_whisper(audio_path=src_path)`, т.е. распознаётся **исходный OGG**, без нашего WAV.
- **Fallback:** при пустом ответе Whisper вызывается Google (`_transcribe_speech_recognition(wav_path, lang)`).
- **Параметры Whisper:** `no_speech_threshold=0.0`, `initial_prompt` для русского, при пустом ответе повтор с `language=None`.

## Что проверить на сервере

1. Запуск с **оригинальным** поведением первого бота (только Google):
   ```bash
   ASR_BACKEND=speech_recognition nohup uvicorn server.main:app --host 0.0.0.0 --port 8001 > server.log 2>&1 &
   ```
   Отправить голосовое и посмотреть `tail -n 80 server.log` — идёт ли запрос в Google и есть ли текст.

2. После обновления кода (с `src_path` для Whisper) — запуск с Whisper:
   ```bash
   ASR_BACKEND=whisper nohup uvicorn server.main:app --host 0.0.0.0 --port 8001 > server.log 2>&1 &
   ```
   В логе должна быть строка вида: `Whisper: using original file /tmp/xxx.ogg (bypass pydub conversion)`. Если после этого появится ненулевой `text_length` — проблема была в конвертации pydub → WAV.

3. Если и с OGG Whisper даёт пусто — значит, либо сам файл с Telegram (битрейт/кодек/тишина), либо окружение Whisper на сервере; тогда остаётся опираться на **speech_recognition** (Google) как основной вариант, как при первом боте.
