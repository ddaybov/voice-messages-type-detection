"""
Inline клавиатуры для Telegram бота.
"""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from ml import get_factory


def get_model_selection_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора модели"""
    factory = get_factory()
    available = factory.get_available_models()

    buttons = []

    sklearn_models = []
    neural_models = []
    other_models = []

    for model_id, info in available.items():
        btn = InlineKeyboardButton(
            text=f"{info['emoji']} {info['name']}",
            callback_data=f"model:{model_id}",
        )

        if info["type"] == "sklearn":
            sklearn_models.append(btn)
        elif info["type"] in ["pytorch", "transformer"]:
            neural_models.append(btn)
        else:
            other_models.append(btn)

    for i in range(0, len(sklearn_models), 2):
        buttons.append(sklearn_models[i : i + 2])

    for i in range(0, len(neural_models), 2):
        buttons.append(neural_models[i : i + 2])

    if other_models:
        buttons.append(other_models)

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_back_keyboard() -> InlineKeyboardMarkup:
    """Кнопка назад"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Выбрать модель", callback_data="select_model")]
        ]
    )
