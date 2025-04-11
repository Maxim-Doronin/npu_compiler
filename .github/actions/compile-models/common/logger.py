"""
Declares and configures loggers
"""

import logging


def setup_thread_file_logger(logs_dir: str, log_file_name: str) -> logging.Logger:
    """Configures and returns a thread-safe file-based logger"""
    logger = logging.getLogger(f"logger_{log_file_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    log_file_path = logs_dir / f"{log_file_name}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def setup_console_logger() -> logging.Logger:
    """Configures and returns a console logger"""
    logger = logging.getLogger("console_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger
