"""
Utility functions for the server.
"""

import os
from typing import Optional
from contextlib import suppress


def safe_unlink(file_path: Optional[str]) -> None:
    """Safely delete a file, ignoring errors."""
    with suppress(Exception):
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)


def cleanup_files(*file_paths: str) -> None:
    """Clean up multiple temporary files."""
    for path in file_paths:
        safe_unlink(path)
