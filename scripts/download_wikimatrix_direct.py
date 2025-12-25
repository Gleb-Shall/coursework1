"""
Попытка прямой загрузки WikiMatrix с OPUS через URL.
"""

import os
import json
import argparse
import urllib.request
import gzip
from tqdm import tqdm


def download_wikimatrix_direct(
    output_file: str = "data/raw/wikimatrix_en_ru.jsonl",
    max_samples: int = None
):
    """
    Пытается загрузить WikiMatrix напрямую с OPUS.
    """
    print("Попытка прямой загрузки WikiMatrix EN-RU с OPUS...")
    print("URL: https://opus.nlpl.eu/WikiMatrix/en&ru/v1/WikiMatrix#download")
    
    # Возможные URL для загрузки (нужно проверить актуальные)
    possible_urls = [
        "https://opus.nlpl.eu/download.php?f=WikiMatrix/v1/moses/en-ru.txt.zip",
        "https://opus.nlpl.eu/download.php?f=WikiMatrix/v1/tmx/en-ru.tmx.zip",
    ]
    
    print("\n⚠ Прямая автоматическая загрузка может не работать.")
    print("Рекомендуется ручная загрузка:")
    print("\n1. Перейдите на:")
    print("   https://opus.nlpl.eu/WikiMatrix/en&ru/v1/WikiMatrix#download")
    print("\n2. Скачайте файлы в формате:")
    print("   - TMX (рекомендуется) ИЛИ")
    print("   - Moses (два файла: .en и .ru)")
    print("\n3. Поместите файлы в data/raw/")
    print("\n4. Конвертируйте:")
    print("   Для TMX:")
    print("   python3 scripts/download_wikimatrix_manual.py \\")
    print("       --tmx_file data/raw/wikimatrix.en-ru.tmx \\")
    print("       --output_file data/raw/wikimatrix_real.jsonl")
    print("\n   Для Moses:")
    print("   python3 scripts/download_wikimatrix_manual.py \\")
    print("       --moses_en data/raw/wikimatrix.en-ru.en \\")
    print("       --moses_ru data/raw/wikimatrix.en-ru.ru \\")
    print("       --output_file data/raw/wikimatrix_real.jsonl")
    
    raise Exception("Прямая загрузка не реализована. Используйте ручную загрузку.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Прямая загрузка WikiMatrix")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/raw/wikimatrix_en_ru.jsonl",
        help="Выходной файл"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Максимальное количество примеров"
    )
    
    args = parser.parse_args()
    
    try:
        download_wikimatrix_direct(args.output_file, args.max_samples)
    except Exception as e:
        print(f"\nОшибка: {e}")

