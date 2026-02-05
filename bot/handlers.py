import os
import logging
import aiohttp
from telegram import Update
from telegram.ext import ContextTypes

from .keyboards import build_model_keyboard, build_back_keyboard

logger = logging.getLogger(__name__)


async def _download_telegram_file(bot, file) -> bytearray:
    """Скачать файл с Telegram по прямому URL (полная загрузка)."""
    try:
        token = getattr(bot, "token", None)
        path = getattr(file, "file_path", None)
        if token and path:
            url = f"https://api.telegram.org/file/bot{token}/{path}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        return bytearray(data)
    except Exception as e:
        logger.warning("Direct Telegram download failed (%s), using download_as_bytearray", e)
    return await file.download_as_bytearray()


USER_MODELS: dict[int, str] = {}
MODELS_CACHE: dict | None = None


async def fetch_models(server_url: str) -> dict:
    global MODELS_CACHE
    if MODELS_CACHE:
        return MODELS_CACHE
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server_url}/models", timeout=10) as resp:
            data = await resp.json()
            MODELS_CACHE = data.get("models", {})
            return MODELS_CACHE


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    USER_MODELS[update.effective_user.id] = "ensemble"
    await update.message.reply_text(
        "👋 Привет! Я бот для классификации голосовых сообщений.\n\n"
        "🎯 Я определяю стиль сообщения: formal или informal\n\n"
        "📝 Команды:\n"
        "/model - выбрать модель классификации\n"
        "/info - информация о текущей модели\n\n"
        "🎤 Отправь голосовое сообщение или текст для классификации!"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    server_url = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")
    models = await fetch_models(server_url)
    current = USER_MODELS.get(update.effective_user.id, "ensemble")
    current_name = models.get(current, {}).get("name", current)
    await update.message.reply_text(
        f"🤖 Текущая модель: {current_name}\n\n"
        "Выберите модель для классификации:",
        reply_markup=build_model_keyboard(models),
    )


async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    server_url = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")
    models = await fetch_models(server_url)
    model_id = USER_MODELS.get(update.effective_user.id, "ensemble")
    info = models.get(model_id, {})
    await update.message.reply_text(
        "📊 Информация о модели\n\n"
        f"🤖 Название: {info.get('name', model_id)}\n"
        f"📁 Категория: {info.get('category', 'unknown')}\n"
        f"📝 Описание: {info.get('description', 'Нет описания')}"
    )


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    if query.data == "noop":
        await query.answer()
        return
    if query.data == "select_model":
        server_url = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")
        models = await fetch_models(server_url)
        await query.message.edit_text(
            "Выберите модель для классификации:",
            reply_markup=build_model_keyboard(models),
        )
        await query.answer()
        return

    if query.data.startswith("model:"):
        model_id = query.data.split(":", 1)[1]
        USER_MODELS[query.from_user.id] = model_id
        server_url = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")
        models = await fetch_models(server_url)
        info = models.get(model_id, {})
        await query.message.edit_text(
            f"✅ Выбрана модель: {info.get('name', model_id)}\n\n"
            f"📝 {info.get('description', '')}\n\n"
            "🎤 Отправьте голосовое сообщение или текст для классификации!",
            reply_markup=build_back_keyboard(),
        )
        await query.answer()


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    model_id = USER_MODELS.get(update.effective_user.id, "ensemble")
    await update.message.reply_text("🔄 Классифицирую текст...")

    server_url = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("text", update.message.text)
            data.add_field("model", model_id)
            async with session.post(f"{server_url}/predict_text", data=data, timeout=30) as resp:
                result = await resp.json()

        if result.get("success"):
            label = result.get("label", "")
            emoji = "👔" if label == "formal" else "😎"
            probs = result.get("probabilities", {})
            await update.message.reply_text(
                f"{emoji} Результат классификации\n\n"
                f"📝 Текст: {update.message.text[:100]}{'...' if len(update.message.text) > 100 else ''}\n\n"
                f"🏷 Класс: {label.upper()}\n"
                f"📊 Уверенность: {result.get('confidence', 0) * 100:.1f}%\n\n"
                f"📈 Вероятности:\n"
                f"  • formal: {probs.get('formal', 0) * 100:.1f}%\n"
                f"  • informal: {probs.get('informal', 0) * 100:.1f}%\n\n"
                f"🤖 Модель: {result.get('model', model_id)}",
                reply_markup=build_back_keyboard(),
            )
        else:
            await update.message.reply_text(f"❌ Ошибка: {result.get('error', 'Unknown error')}")
    except Exception as exc:
        await update.message.reply_text(f"❌ Ошибка: {exc}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    model_id = USER_MODELS.get(update.effective_user.id, "ensemble")
    await update.message.reply_text("🔄 Обрабатываю голосовое сообщение...")

    server_url = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")
    try:
        if update.message.voice:
            tg_voice = update.message.voice
            file = await context.bot.get_file(tg_voice.file_id)
            tg_size = getattr(tg_voice, "file_size", None)
            tg_mime = getattr(tg_voice, "mime_type", None)
        elif update.message.audio:
            tg_audio = update.message.audio
            file = await context.bot.get_file(tg_audio.file_id)
            tg_size = getattr(tg_audio, "file_size", None)
            tg_mime = getattr(tg_audio, "mime_type", None)
        else:
            await update.message.reply_text("❌ Неизвестный формат аудио")
            return

        # Расширение из file_path Telegram (голос = OGG Opus, часто .oga). См. https://core.telegram.org/bots/api#voice
        file_path = getattr(file, "file_path", None)
        ext = os.path.splitext(file_path)[1] if file_path else ".oga"
        if ext.lower() not in (".oga", ".ogg", ".opus"):
            ext = ".oga"
        filename = f"audio{ext}"
        content_type = tg_mime or "audio/ogg"

        # Скачиваем по прямому URL, чтобы гарантированно получить полный файл
        file_bytes = await _download_telegram_file(context.bot, file)
        logger.info(
            "Voice/audio downloaded: %d bytes (Telegram file_size=%s), filename=%s, sending to %s/predict",
            len(file_bytes), tg_size, filename, server_url,
        )
        if len(file_bytes) == 0:
            await update.message.reply_text("❌ Не удалось загрузить файл (0 байт)")
            return

        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("file", file_bytes, filename=filename, content_type=content_type)
            data.add_field("lang", "ru-RU")
            data.add_field("model", model_id)
            async with session.post(f"{server_url}/predict", data=data, timeout=120) as resp:
                result = await resp.json()
                logger.info("Predict response: success=%s, status=%d", result.get("success"), resp.status)

        if result.get("success"):
            label_name = result.get("label_name", "")
            emoji = "👔" if label_name == "formal" else "😎"
            await update.message.reply_text(
                f"{emoji} Результат классификации\n\n"
                f"📝 Текст: {result.get('text', '')[:100]}\n\n"
                f"🏷 Класс: {label_name.upper()}\n"
                f"📊 Уверенность: {result.get('confidence', 0) * 100:.1f}%\n\n"
                f"⏱ Длительность: {result.get('duration', 0):.1f} сек\n"
                f"🤖 Модель: {result.get('model', model_id)}",
                reply_markup=build_back_keyboard(),
            )
        else:
            await update.message.reply_text(f"❌ Ошибка: {result.get('error', 'Unknown error')}")
    except Exception as exc:
        await update.message.reply_text(f"❌ Ошибка: {exc}")
