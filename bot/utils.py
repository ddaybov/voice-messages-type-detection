"""
Utility functions for Telegram bot.
"""

from typing import Optional
from io import BytesIO
import logging

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def download_file(
    file_id: str, context: ContextTypes.DEFAULT_TYPE
) -> Optional[BytesIO]:
    """Download file from Telegram."""
    try:
        file = await context.bot.get_file(file_id)
        file_bytes = BytesIO()
        await file.download_to_memory(file_bytes)
        file_bytes.seek(0)
        return file_bytes
    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {e}")
        return None


def truncate_message(text: str, max_length: int = 4096) -> str:
    """Truncate message to Telegram's limit."""
    if len(text) > max_length:
        return text[:max_length - 3] + "..."
    return text
