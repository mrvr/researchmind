"""
utils/logger.py — Structured, coloured console logger for ResearchMind.

Usage:
    from utils.logger import get_logger
    log = get_logger("MyModule")
    log.info("Processing started")
    log.warning("Something looks off")
    log.error("Something failed")
"""

import logging
import sys

# ── ANSI colour codes ──────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
GREY    = "\033[90m"
WHITE   = "\033[97m"


class _ColorFormatter(logging.Formatter):
    """Custom formatter that adds colour to log level names and logger names."""

    LEVEL_COLORS = {
        logging.DEBUG:    GREY,
        logging.INFO:     CYAN,
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: BOLD + RED,
    }

    LEVEL_ICONS = {
        logging.DEBUG:    "·",
        logging.INFO:     "●",
        logging.WARNING:  "▲",
        logging.ERROR:    "✗",
        logging.CRITICAL: "✗✗",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        icon  = self.LEVEL_ICONS.get(record.levelno, " ")

        # Coloured level label — fixed width for alignment
        record.levelname = f"{color}{icon} {record.levelname:<8}{RESET}"

        # Dimmed logger name in brackets
        record.name = f"{GREY}[{record.name}]{RESET}"

        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get (or create) a named logger with coloured console output.

    Args:
        name:  Logger name shown in output — use your module/class name
        level: Logging level (default: INFO)

    Returns:
        Configured logging.Logger instance

    Example:
        log = get_logger("AudioProcessor")
        log.info("Transcribing audio...")
        log.warning("Low VRAM detected")
        log.error("FFmpeg not found")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColorFormatter(
            fmt="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False   # Don't bubble up to root logger

    return logger


def set_level(name: str, level: int):
    """Change the log level of an existing logger at runtime."""
    logging.getLogger(name).setLevel(level)


def silence(name: str):
    """Suppress all output from a specific logger (e.g. noisy third-party libs)."""
    logging.getLogger(name).setLevel(logging.CRITICAL + 1)
