"""
Скрипт предобработки данных для машинного перевода математических текстов.
"""

import os
import json
import re
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.formula_handler import FormulaHandler


def normalize_text(text: str) -> str:
    """Нормализация текста."""
    # Нормализация пробелов
    text = re.sub(r'\s+', ' ', text)
    # Удаление лишних пробелов вокруг формул
    text = re.sub(r'\s+\$', '$', text)
    text = re.sub(r'\$\s+', '$', text)
    text = text.strip()
    return text


def is_valid_pair(source: str, target: str, 
                 min_length: int = 5, 
                 max_length: int = 512) -> bool:
    """Проверяет валидность пары предложений."""
    # Проверка длины
    if len(source.split()) < min_length or len(target.split()) < min_length:
        return False
    if len(source) > max_length or len(target) > max_length:
        return False
    
    # Проверка на пустые строки
    if not source.strip() or not target.strip():
        return False
    
    # Проверка на слишком большое расхождение в длине
    len_ratio = len(target) / len(source) if len(source) > 0 else 0
    if len_ratio < 0.3 or len_ratio > 3.0:
        return False
    
    return True


def preprocess_dataset(
    input_file: str,
    output_file: str,
    formula_handler: FormulaHandler,
    filter_math: bool = True,
    min_length: int = 5,
    max_length: int = 512,
    preserve_formulas: bool = True
) -> Tuple[int, int]:
    """
    Предобрабатывает датасет.
    
    Returns:
        (total_count, processed_count)
    """
    print(f"Чтение данных из {input_file}...")
    
    processed_pairs = []
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Обработка"):
            total_count += 1
            try:
                data = json.loads(line.strip())
                source = data.get("source", "").strip()
                target = data.get("target", "").strip()
                
                if not source or not target:
                    continue
                
                # Нормализация
                source = normalize_text(source)
                target = normalize_text(target)
                
                # Проверка валидности
                if not is_valid_pair(source, target, min_length, max_length):
                    continue
                
                # Фильтрация математических текстов (строгая)
                if filter_math:
                    # Проверяем все три критерия: формулы, ключевые слова, паттерны
                    is_math = (
                        formula_handler.has_formula(source) or 
                        formula_handler.has_math_keywords(source) or
                        formula_handler.has_math_patterns(source)
                    )
                    if not is_math:
                        continue
                
                # Обработка формул
                if preserve_formulas:
                    # Проверяем валидность формул
                    source_formulas = formula_handler.find_formulas(source)
                    target_formulas = formula_handler.find_formulas(target)
                    
                    # Пропускаем если есть битые формулы
                    if any(not formula_handler.is_valid_formula(f[0]) 
                          for f in source_formulas + target_formulas):
                        continue
                
                processed_pairs.append({
                    "source": source,
                    "target": target
                })
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Ошибка при обработке строки: {e}")
                continue
    
    print(f"Обработано {len(processed_pairs)} из {total_count} примеров")
    
    # Сохранение
    print(f"Сохранение в {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in processed_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write('\n')
    
    return total_count, len(processed_pairs)


def split_dataset(
    input_file: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05
):
    """Разделяет датасет на train/val/test."""
    print("Чтение данных для разделения...")
    
    pairs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    
    print(f"Всего примеров: {len(pairs)}")
    
    # Перемешиваем
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    # Разделяем
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = pairs[:train_end]
    val_data = pairs[train_end:val_end]
    test_data = pairs[val_end:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Сохраняем
    base_dir = os.path.dirname(input_file)
    
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_file = os.path.join(base_dir, f"{split_name}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in split_data:
                json.dump(pair, f, ensure_ascii=False)
                f.write('\n')
        print(f"Сохранён {split_name}: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Предобработка датасета")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/raw/wikimatrix_en_ru.jsonl",
        help="Входной файл (JSONL)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/preprocessed.jsonl",
        help="Выходной файл (JSONL)"
    )
    parser.add_argument(
        "--filter_math",
        action="store_true",
        default=True,
        help="Фильтровать только математические тексты"
    )
    parser.add_argument(
        "--no_filter_math",
        dest="filter_math",
        action="store_false",
        help="Не фильтровать математические тексты"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Минимальная длина предложения"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Максимальная длина предложения"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Разделить на train/val/test после предобработки"
    )
    
    args = parser.parse_args()
    
    formula_handler = FormulaHandler(preserve_formulas=True)
    
    total, processed = preprocess_dataset(
        args.input_file,
        args.output_file,
        formula_handler,
        filter_math=args.filter_math,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    print(f"\nСтатистика:")
    print(f"Всего примеров: {total}")
    print(f"Обработано: {processed}")
    print(f"Процент: {100 * processed / total:.2f}%")
    
    if args.split:
        split_dataset(args.output_file)


if __name__ == "__main__":
    main()

