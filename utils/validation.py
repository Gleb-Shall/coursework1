"""
Утилиты для валидации данных и конфигурации.
"""

import os
import json
from typing import List, Dict, Tuple, Optional


def validate_data_file(file_path: str) -> Tuple[bool, List[str]]:
    """
    Валидирует файл с данными.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if not os.path.exists(file_path):
        return False, [f"Файл не найден: {file_path}"]
    
    if not file_path.endswith('.jsonl'):
        errors.append("Файл должен иметь расширение .jsonl")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for i, line in enumerate(f, 1):
                line_count += 1
                try:
                    data = json.loads(line.strip())
                    if not isinstance(data, dict):
                        errors.append(f"Строка {i}: данные должны быть словарём")
                        continue
                    
                    if "source" not in data or "target" not in data:
                        errors.append(f"Строка {i}: отсутствуют поля 'source' или 'target'")
                        continue
                    
                    if not isinstance(data["source"], str) or not isinstance(data["target"], str):
                        errors.append(f"Строка {i}: 'source' и 'target' должны быть строками")
                        continue
                    
                    if not data["source"].strip() or not data["target"].strip():
                        errors.append(f"Строка {i}: пустые 'source' или 'target'")
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Строка {i}: ошибка JSON - {str(e)}")
            
            if line_count == 0:
                errors.append("Файл пуст")
    
    except Exception as e:
        return False, [f"Ошибка при чтении файла: {str(e)}"]
    
    return len(errors) == 0, errors


def validate_model_dir(model_dir: str) -> Tuple[bool, List[str]]:
    """
    Валидирует директорию с моделью.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if not os.path.exists(model_dir):
        return False, [f"Директория не найдена: {model_dir}"]
    
    required_files = ["config.json", "pytorch_model.bin"]
    # Также может быть safetensors
    has_model_file = (
        os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) or
        os.path.exists(os.path.join(model_dir, "model.safetensors"))
    )
    
    if not has_model_file:
        errors.append("Не найден файл модели (pytorch_model.bin или model.safetensors)")
    
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        errors.append("Не найден config.json")
    
    return len(errors) == 0, errors


def validate_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Валидирует конфигурацию обучения.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    required_keys = ["batch_size", "learning_rate", "num_epochs"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Отсутствует обязательный параметр: {key}")
    
    if "batch_size" in config:
        if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
            errors.append("batch_size должен быть положительным целым числом")
    
    if "learning_rate" in config:
        if not isinstance(config["learning_rate"], (int, float)) or config["learning_rate"] <= 0:
            errors.append("learning_rate должен быть положительным числом")
    
    if "num_epochs" in config:
        if not isinstance(config["num_epochs"], int) or config["num_epochs"] <= 0:
            errors.append("num_epochs должен быть положительным целым числом")
    
    return len(errors) == 0, errors


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Проверяет наличие необходимых зависимостей.
    
    Returns:
        (all_installed, list_of_missing)
    """
    missing = []
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "sacrebleu",
        "numpy",
        "tqdm"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing


def check_gpu() -> Dict[str, any]:
    """
    Проверяет наличие и характеристики GPU.
    
    Returns:
        Словарь с информацией о GPU
    """
    import torch
    
    info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
        "memory_total": None,
        "memory_allocated": None
    }
    
    if info["available"]:
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(info["current_device"])
        info["memory_total"] = torch.cuda.get_device_properties(
            info["current_device"]
        ).total_memory / (1024**3)  # GB
        info["memory_allocated"] = torch.cuda.memory_allocated(
            info["current_device"]
        ) / (1024**3)  # GB
    
    return info

