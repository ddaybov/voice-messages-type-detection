"""
Telegram bot for voice messages type detection.
Accepts voice messages and audio files, sends them to FastAPI server for classification.
"""

import os
import sys
import logging
from typing import Optional

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Use absolute imports that work both as script and module
try:
    from telegram.config import (
        API_URL, BOT_TOKEN, DEFAULT_MODEL, DEFAULT_LANG,
        AVAILABLE_MODELS, user_sessions, SERVER_URL
    )
    from telegram.utils import download_file, truncate_message
except ImportError:
    # Fallback for relative imports (when run as module)
    from .config import (
        API_URL, BOT_TOKEN, DEFAULT_MODEL, DEFAULT_LANG,
        AVAILABLE_MODELS, user_sessions, SERVER_URL
    )
    from .utils import download_file, truncate_message

try:
    from server.constants import DEFAULT_MAX_TEXT_PREVIEW_LENGTH, DEFAULT_CONFIDENCE_BAR_LENGTH
except ImportError:
    # Fallback values if constants not available
    DEFAULT_MAX_TEXT_PREVIEW_LENGTH = 100
    DEFAULT_CONFIDENCE_BAR_LENGTH = 10

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    if not update.message:
        return
    
    user_id = update.effective_user.id
    # Use logreg as fallback if default model is not available
    user_sessions[user_id] = (
        DEFAULT_MODEL if DEFAULT_MODEL in AVAILABLE_MODELS else "logreg"
    )

    model_name = AVAILABLE_MODELS.get(user_sessions[user_id], user_sessions[user_id])
    welcome_text = (
        "🎤 <b>Бот для определения типа голосовых сообщений</b>\n\n"
        "Отправьте голосовое сообщение или аудиофайл, и я определю, "
        "является ли оно формальным или неформальным.\n\n"
        f"<b>Текущая модель:</b> {model_name}\n\n"
        "Доступные команды:\n"
        "/start - Начать работу\n"
        "/model - Выбрать модель\n"
        "/help - Справка"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = (
        "📖 <b>Справка</b>\n\n"
        "Этот бот классифицирует голосовые сообщения на формальные и неформальные.\n\n"
        "<b>Как использовать:</b>\n"
        "1. Отправьте голосовое сообщение или аудиофайл\n"
        "2. Бот автоматически распознает речь и определит тип\n"
        "3. Вы получите результат с уверенностью и превью текста\n\n"
        "<b>Поддерживаемые форматы:</b>\n"
        "• Голосовые сообщения Telegram\n"
        "• WAV, MP3, M4A, AAC, FLAC, OGG, WMA\n\n"
        "<b>Команды:</b>\n"
        "/start - Начать работу\n"
        "/model - Выбрать модель классификации\n"
        "/status - Проверить статус сервера\n"
        "/help - Показать эту справку"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command - check server availability."""
    status_msg = await update.message.reply_text("🔍 Проверяю статус сервера...")
    
    # Check server health
    is_healthy = await check_server_health()
    
    if is_healthy:
        # Try to get supported formats
        try:
            base_url = SERVER_URL.replace("/predict", "").rstrip("/")
            formats_url = f"{base_url}/supported_formats"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    formats_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        formats = ", ".join(data.get("audio_formats", []))
                        languages = ", ".join(data.get("languages", {}).keys())
                        
                        status_text = (
                            "✅ <b>Сервер работает</b>\n\n"
                            f"🌐 URL: <code>{SERVER_URL}</code>\n"
                            f"📦 Форматы: {formats}\n"
                            f"🌍 Языки: {languages}\n"
                            f"🤖 Модель по умолчанию: {DEFAULT_MODEL}"
                        )
                    else:
                        status_text = (
                            "✅ <b>Сервер работает</b>\n\n"
                            f"🌐 URL: <code>{SERVER_URL}</code>\n"
                            f"⚠️ Не удалось получить информацию о форматах"
                        )
        except Exception as e:
            logger.exception("Error getting server info: %s", e)
            status_text = (
                "✅ <b>Сервер работает</b>\n\n"
                f"🌐 URL: <code>{SERVER_URL}</code>\n"
                f"⚠️ Не удалось получить дополнительную информацию"
            )
    else:
        status_text = (
            "❌ <b>Сервер недоступен</b>\n\n"
            f"🌐 URL: <code>{SERVER_URL}</code>\n\n"
            "💡 Убедитесь, что сервер запущен:\n"
            "<code>uvicorn server.main:app --host 0.0.0.0 --port 8000</code>"
        )
    
    await status_msg.edit_text(status_text, parse_mode="HTML")


async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model command - show model selection keyboard."""
    keyboard = []
    row = []
    for i, (model_id, model_name) in enumerate(AVAILABLE_MODELS.items()):
        row.append(
            InlineKeyboardButton(
                model_name,
                callback_data=f"model_{model_id}",
            )
        )
        if (i + 1) % 2 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    reply_markup = InlineKeyboardMarkup(keyboard)
    user_id = update.effective_user.id
    current_model = user_sessions.get(user_id, DEFAULT_MODEL)

    text = (
        f"🔧 <b>Выбор модели</b>\n\n"
        f"Текущая модель: <b>{AVAILABLE_MODELS.get(current_model, current_model)}</b>\n\n"
        "Выберите модель для классификации:"
    )
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle model selection callback."""
    query = update.callback_query
    await query.answer()

    if not query.data or not query.data.startswith("model_"):
        return

    model_id = query.data.replace("model_", "")
    user_id = update.effective_user.id
    user_sessions[user_id] = model_id

    model_name = AVAILABLE_MODELS.get(model_id, model_id)
    await query.edit_message_text(
        f"✅ Модель изменена на: <b>{model_name}</b>",
        parse_mode="HTML",
    )




async def check_server_health() -> bool:
    """Check if server is available."""
    # Build health URL from SERVER_URL
    base_url = SERVER_URL.replace("/predict", "").rstrip("/")
    health_url = f"{base_url}/health"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                health_url,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                return response.status == 200
    except Exception:
        return False


async def send_to_api(
    audio_bytes,
    filename: str,
    model: str,
    lang: str = DEFAULT_LANG,
) -> Optional[dict]:
    """Send audio file to FastAPI server for prediction."""
    # Check server health first
    if not await check_server_health():
        return {
            "success": False,
            "error": "Сервер недоступен. Убедитесь, что сервер запущен (uvicorn server.main:app).",
        }
    
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("file", audio_bytes, filename=filename)
            data.add_field("lang", lang)
            data.add_field("model", model)

            async with session.post(
                API_URL,
                data=data,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("detail", error_data.get("error", f"HTTP {response.status}"))
                    except:
                        error_text = await response.text()
                        error_msg = error_text if error_text else f"HTTP {response.status}"
                    logger.error(
                        f"API error {response.status}: {error_msg}",
                    )
                    return {"success": False, "error": error_msg}
    except aiohttp.ClientError as e:
        logger.error(f"Connection error: {e}")
        return {"success": False, "error": "Не удалось подключиться к серверу"}
    except Exception as e:
        logger.exception(f"Error sending to API: {e}")
        return {"success": False, "error": str(e)}


def format_response(result: dict) -> str:
    """Format API response for Telegram message."""
    if not result.get("success"):
        error = result.get("error", "Неизвестная ошибка")
        
        # More user-friendly error messages
        error_messages = {
            "Empty text": "Не удалось распознать текст из аудио. Проверьте качество записи или настройки ASR.",
            "Empty text for DimaNet": "Не удалось распознать текст из аудио для DimaNet.",
        }
        
        # Check if error starts with known patterns
        for key, friendly_msg in error_messages.items():
            if key in error:
                error = friendly_msg
                break
        
        # Add helpful tips for common errors
        if "Unknown model" in error:
            error += "\n\n💡 Используйте /model для выбора доступной модели."
        elif "Empty text" in error or "распознать текст" in error:
            error += "\n\n💡 Попробуйте:\n• Улучшить качество записи\n• Говорить четче\n• Проверить настройки ASR в .env"
        elif "not found" in error.lower() or "не найден" in error.lower():
            error += "\n\n💡 Убедитесь, что модель обучена и находится в папке models/"
        
        return f"❌ <b>Ошибка:</b> {error}"

    label_name = result.get("label_name", "unknown")
    confidence = result.get("confidence", 0.0)
    text = result.get("text", "")
    duration = result.get("duration", 0.0)
    word_count = result.get("word_count", 0)
    model = result.get("model", "unknown")
    asr_backend = result.get("asr_backend", "unknown")

    # Emoji for label
    label_emoji = "📋" if label_name == "formal" else "💬"
    label_display = "Формальный" if label_name == "formal" else "Неформальный"

    # Confidence bar
    conf_percent = int(confidence * 100)
    filled = int(conf_percent / 100 * DEFAULT_CONFIDENCE_BAR_LENGTH)
    conf_bar = "█" * filled + "░" * (DEFAULT_CONFIDENCE_BAR_LENGTH - filled)

    # Preview text
    if len(text) > DEFAULT_MAX_TEXT_PREVIEW_LENGTH:
        text_preview = text[:DEFAULT_MAX_TEXT_PREVIEW_LENGTH] + "..."
    else:
        text_preview = text

    # Format response according to README requirements:
    # - Класс (class)
    # - Уверенность (confidence)
    # - Длительность (duration)
    # - Число слов (word count)
    # - Превью текста (text preview)
    response_text = (
        f"{label_emoji} <b>Результат классификации</b>\n\n"
        f"📋 <b>Класс:</b> {label_display}\n"
        f"📊 <b>Уверенность:</b> {conf_percent}% {conf_bar}\n"
        f"⏱️ <b>Длительность:</b> {duration:.1f} сек\n"
        f"📝 <b>Число слов:</b> {word_count}\n\n"
        f"💬 <b>Превью текста:</b>\n"
        f"<i>{text_preview}</i>\n\n"
        f"<code>Модель: {model} | ASR: {asr_backend}</code>"
    )

    return response_text


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages."""
    if not update.message or not update.message.voice:
        return

    user_id = update.effective_user.id
    model = user_sessions.get(user_id, DEFAULT_MODEL)
    # Fallback to logreg if model not in available models
    if model not in AVAILABLE_MODELS:
        model = "logreg"
        user_sessions[user_id] = model

    # Send processing message
    processing_msg = await update.message.reply_text("🔄 Обрабатываю голосовое сообщение...")

    # Download voice file
    voice = update.message.voice
    file_bytes = await download_file(voice.file_id, context)
    if not file_bytes:
        await processing_msg.edit_text("❌ Ошибка при загрузке файла")
        return

    # Send to API
    result = await send_to_api(
        file_bytes,
        filename=f"voice_{voice.file_id}.ogg",
        model=model,
    )

    if not result:
        await processing_msg.edit_text("❌ Ошибка при отправке на сервер")
        return

    # Format and send response
    response_text = format_response(result)
    response_text = truncate_message(response_text, max_length=4096)
    await processing_msg.edit_text(response_text, parse_mode="HTML")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio files."""
    if not update.message or not update.message.audio:
        return

    user_id = update.effective_user.id
    model = user_sessions.get(user_id, DEFAULT_MODEL)
    # Fallback to logreg if model not in available models
    if model not in AVAILABLE_MODELS:
        model = "logreg"
        user_sessions[user_id] = model

    # Send processing message
    processing_msg = await update.message.reply_text("🔄 Обрабатываю аудиофайл...")

    # Download audio file
    audio = update.message.audio
    file_bytes = await download_file(audio.file_id, context)
    if not file_bytes:
        await processing_msg.edit_text("❌ Ошибка при загрузке файла")
        return

    # Get filename
    filename = audio.file_name or f"audio_{audio.file_id}.mp3"

    # Send to API
    result = await send_to_api(
        file_bytes,
        filename=filename,
        model=model,
    )

    if not result:
        await processing_msg.edit_text("❌ Ошибка при отправке на сервер")
        return

    # Format and send response
    response_text = format_response(result)
    response_text = truncate_message(response_text, max_length=4096)
    await processing_msg.edit_text(response_text, parse_mode="HTML")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document files (audio files sent as documents)."""
    if not update.message or not update.message.document:
        return

    document = update.message.document
    mime_type = document.mime_type or ""

    # Check if it's an audio file
    if not mime_type.startswith("audio/"):
        return

    user_id = update.effective_user.id
    model = user_sessions.get(user_id, DEFAULT_MODEL)
    # Fallback to logreg if model not in available models
    if model not in AVAILABLE_MODELS:
        model = "logreg"
        user_sessions[user_id] = model

    # Send processing message
    processing_msg = await update.message.reply_text("🔄 Обрабатываю аудиофайл...")

    # Download document
    file_bytes = await download_file(document.file_id, context)
    if not file_bytes:
        await processing_msg.edit_text("❌ Ошибка при загрузке файла")
        return

    # Get filename
    filename = document.file_name or f"audio_{document.file_id}"

    # Send to API
    result = await send_to_api(
        file_bytes,
        filename=filename,
        model=model,
    )

    if not result:
        await processing_msg.edit_text("❌ Ошибка при отправке на сервер")
        return

    # Format and send response
    response_text = format_response(result)
    response_text = truncate_message(response_text, max_length=4096)
    await processing_msg.edit_text(response_text, parse_mode="HTML")


def main() -> None:
    """Start the bot."""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN environment variable is not set!")
        return

    # Create application
    application = Application.builder().token(BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("model", model_selection))
    application.add_handler(CallbackQueryHandler(model_callback, pattern="^model_"))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    application.add_handler(MessageHandler(filters.Document.AUDIO, handle_document))

    # Start bot
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
