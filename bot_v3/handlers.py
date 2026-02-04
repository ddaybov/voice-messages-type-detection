"""
Обработчики сообщений бота.
"""

from io import BytesIO
import os
import aiohttp
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from ml import get_factory
from .keyboards import get_model_selection_keyboard, get_back_keyboard

router = Router()

user_models: dict[int, str] = {}


def get_user_model(user_id: int) -> str:
    factory = get_factory()
    return user_models.get(user_id, factory.get_default_model())


@router.message(Command("start"))
async def cmd_start(message: Message):
    factory = get_factory()
    user_models[message.from_user.id] = factory.get_default_model()

    await message.answer(
        "👋 Привет! Я определяю стиль текста (формальный/неформальный).\n\n"
        "📝 Отправь текстовое или голосовое сообщение.\n"
        "⚙️ /model — выбрать модель классификации\n"
        "ℹ️ /info — информация о текущей модели"
    )


@router.message(Command("model"))
async def cmd_model(message: Message):
    await message.answer(
        "🤖 Выберите модель классификации:", reply_markup=get_model_selection_keyboard()
    )


@router.message(Command("info"))
async def cmd_info(message: Message):
    factory = get_factory()
    model_id = get_user_model(message.from_user.id)
    info = factory.MODEL_INFO.get(model_id, {})

    await message.answer(
        f"{info.get('emoji', '🤖')} **{info.get('name', model_id)}**\n\n"
        f"📝 {info.get('description', 'Нет описания')}\n"
        f"🏷 Тип: {info.get('type', 'unknown')}",
        parse_mode="Markdown",
    )


@router.callback_query(F.data.startswith("model:"))
async def callback_model_select(callback: CallbackQuery):
    model_id = callback.data.split(":", 1)[1]
    user_models[callback.from_user.id] = model_id

    factory = get_factory()
    info = factory.MODEL_INFO.get(model_id, {})

    await callback.message.edit_text(
        f"✅ Выбрана модель: {info.get('emoji', '')} **{info.get('name', model_id)}**\n\n"
        f"📝 {info.get('description', '')}\n\n"
        f"Теперь отправьте текст или голосовое сообщение для классификации.",
        parse_mode="Markdown",
        reply_markup=get_back_keyboard(),
    )
    await callback.answer()


@router.callback_query(F.data == "select_model")
async def callback_back_to_models(callback: CallbackQuery):
    await callback.message.edit_text(
        "🤖 Выберите модель классификации:",
        reply_markup=get_model_selection_keyboard(),
    )
    await callback.answer()


@router.message(F.text)
async def handle_text(message: Message):
    factory = get_factory()
    model_id = get_user_model(message.from_user.id)

    try:
        model = factory.get_model(model_id)
        model.ensure_loaded()

        label, confidence = model.predict(message.text)
        proba = model.predict_proba(message.text)

        emoji = "👔" if label == "formal" else "😎"
        info = factory.MODEL_INFO.get(model_id, {})

        await message.answer(
            f"{emoji} **{label.upper()}**\n\n"
            f"📊 Уверенность: {confidence:.1%}\n"
            f"📈 formal: {proba['formal']:.1%} | informal: {proba['informal']:.1%}\n\n"
            f"🤖 Модель: {info.get('emoji', '')} {info.get('name', model_id)}",
            parse_mode="Markdown",
            reply_markup=get_back_keyboard(),
        )
    except Exception as e:
        await message.answer(f"❌ Ошибка: {e}")


@router.message(F.voice | F.audio)
async def handle_voice(message: Message):
    model_id = get_user_model(message.from_user.id)
    await message.answer("🔄 Обрабатываю голосовое сообщение...")

    server_url = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")
    try:
        if message.voice:
            file = await message.bot.get_file(message.voice.file_id)
        else:
            file = await message.bot.get_file(message.audio.file_id)

        buffer = BytesIO()
        await message.bot.download_file(file.file_path, buffer)
        buffer.seek(0)

        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("file", buffer.read(), filename="audio.ogg")
            data.add_field("model", model_id)
            async with session.post(f"{server_url}/predict", data=data, timeout=120) as resp:
                result = await resp.json()

        if result.get("success"):
            label = result.get("label_name", result.get("label", ""))
            emoji = "👔" if label == "formal" else "😎"
            await message.answer(
                f"{emoji} **{label.upper()}**\n\n"
                f"📊 Уверенность: {result.get('confidence', 0) * 100:.1f}%\n"
                f"📝 Текст: {result.get('text', '')[:120]}\n",
                parse_mode="Markdown",
                reply_markup=get_back_keyboard(),
            )
        else:
            await message.answer(f"❌ Ошибка: {result.get('error', 'Unknown error')}")
    except Exception as e:
        await message.answer(f"❌ Ошибка: {e}")
