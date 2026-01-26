from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def build_model_keyboard(models: dict) -> InlineKeyboardMarkup:
    rows = []

    trained = [(k, v) for k, v in models.items() if v.get("category") == "trained"]
    pretrained = [(k, v) for k, v in models.items() if v.get("category") == "pretrained"]

    if trained:
        rows.append([InlineKeyboardButton("📚 Обученные модели", callback_data="noop")])
        for model_id, info in trained:
            text = f"{info.get('emoji', '🤖')} {info.get('name', model_id)}"
            rows.append([InlineKeyboardButton(text, callback_data=f"model:{model_id}")])

    if pretrained:
        rows.append([InlineKeyboardButton("🌐 Pretrained модели", callback_data="noop")])
        for model_id, info in pretrained:
            text = f"{info.get('emoji', '🤖')} {info.get('name', model_id)}"
            rows.append([InlineKeyboardButton(text, callback_data=f"model:{model_id}")])

    return InlineKeyboardMarkup(rows)


def build_back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("◀️ Выбрать другую модель", callback_data="select_model")]]
    )
