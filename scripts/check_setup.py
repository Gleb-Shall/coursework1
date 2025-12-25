"""
Скрипт для проверки готовности окружения к запуску проекта.
"""

import os
import sys

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.validation import check_dependencies, check_gpu, validate_data_file


def main():
    print("="*60)
    print("ПРОВЕРКА ОКРУЖЕНИЯ")
    print("="*60)
    
    # Проверка зависимостей
    print("\n1. Проверка зависимостей...")
    all_installed, missing = check_dependencies()
    if all_installed:
        print("   ✓ Все необходимые пакеты установлены")
    else:
        print(f"   ✗ Отсутствуют пакеты: {', '.join(missing)}")
        print("   Установите их командой: pip install -r requirements.txt")
    
    # Проверка GPU
    print("\n2. Проверка GPU...")
    gpu_info = check_gpu()
    if gpu_info["available"]:
        print(f"   ✓ GPU доступен: {gpu_info['device_name']}")
        print(f"   Память: {gpu_info['memory_total']:.2f} GB (доступно)")
        if gpu_info['memory_allocated'] > 0:
            print(f"   Используется: {gpu_info['memory_allocated']:.2f} GB")
    else:
        print("   ⚠ GPU не обнаружен, будет использоваться CPU")
        print("   Обучение может быть очень медленным")
    
    # Проверка структуры проекта
    print("\n3. Проверка структуры проекта...")
    required_dirs = ["data/raw", "data/processed", "models", "scripts", "utils", "results"]
    all_dirs_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✓ {dir_path}")
        else:
            print(f"   ✗ {dir_path} - не найдена")
            all_dirs_exist = False
    
    if not all_dirs_exist:
        print("\n   Создайте недостающие директории")
    
    # Проверка данных (если есть)
    print("\n4. Проверка данных...")
    data_files = [
        "data/raw/wikimatrix_en_ru.jsonl",
        "data/processed/preprocessed.jsonl",
        "data/processed/train.jsonl",
        "data/processed/val.jsonl",
        "data/processed/test.jsonl"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            is_valid, errors = validate_data_file(data_file)
            if is_valid:
                # Подсчёт строк
                with open(data_file, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                print(f"   ✓ {data_file} ({count} примеров)")
            else:
                print(f"   ✗ {data_file} - ошибки:")
                for error in errors[:3]:  # Показываем первые 3 ошибки
                    print(f"      - {error}")
        else:
            print(f"   - {data_file} - не найден (это нормально, если данные ещё не загружены)")
    
    print("\n" + "="*60)
    if all_installed and all_dirs_exist:
        print("✓ Окружение готово к работе!")
    else:
        print("⚠ Некоторые проверки не пройдены, исправьте ошибки выше")
    print("="*60)


if __name__ == "__main__":
    main()

