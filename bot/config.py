"""
Configuration for Telegram bot.
"""

import os

# Server configuration
# If SERVER_URL is not set, try to construct it from HOST and PORT
default_port = os.getenv("PORT", "8000")
default_host = os.getenv("HOST", "127.0.0.1")
default_server_url = f"http://{default_host}:{default_port}"

SERVER_URL = os.getenv("SERVER_URL", default_server_url)
if not SERVER_URL.endswith("/predict"):
    API_URL = f"{SERVER_URL.rstrip('/')}/predict"
else:
    API_URL = SERVER_URL

# Bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "logreg")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "ru-RU")

# Available models
AVAILABLE_MODELS = {
    "daybovnet": "DaybovNet (авторская CNN)",
    "logreg": "Logistic Regression",
    "svm": "SVM",
    "nb": "Naive Bayes",
    "bert": "BERT Transformer",
    "dimanet": "DimaNet (CNN+BiLSTM)",
    "ensemble": "Ансамбль моделей",
}

# User session storage: user_id -> selected_model
user_sessions = {}
