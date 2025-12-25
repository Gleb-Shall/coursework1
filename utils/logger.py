"""
Утилита для логирования в проекте.
"""

import logging
import os
from datetime import datetime


def setup_logger(
    name: str = "translation_project",
    log_dir: str = "logs",
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Настраивает логгер для проекта.
    
    Args:
        name: Имя логгера
        log_dir: Директория для логов
        level: Уровень логирования
        log_to_file: Сохранять ли логи в файл
    
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Очищаем существующие handlers
    logger.handlers.clear()
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

