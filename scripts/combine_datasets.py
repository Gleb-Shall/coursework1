"""
Скрипт для объединения нескольких датасетов.
Позволяет комбинировать базовый датасет с математическими примерами.
"""

import os
import json
import argparse
import random


def combine_datasets(
    input_files: list,
    output_file: str,
    math_ratio: float = 0.3,
    max_total: int = None
):
    """
    Объединяет датасеты, сохраняя пропорцию математических текстов.
    
    Args:
        input_files: Список файлов для объединения
        output_file: Выходной файл
        math_ratio: Доля математических текстов (0-1)
        max_total: Максимальное количество примеров (None = все)
    """
    all_pairs = []
    math_pairs = []
    other_pairs = []
    
    # Загружаем все данные
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Пропускаем несуществующий файл: {input_file}")
            continue
        
        print(f"Загрузка {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pair = json.loads(line.strip())
                    if 'source' in pair and 'target' in pair:
                        # Определяем, математический ли это текст
                        source = pair['source'].lower()
                        is_math = any(kw in source for kw in [
                            '$', 'theorem', 'proof', 'derivative', 'integral',
                            'matrix', 'equation', 'function', 'теорема',
                            'производная', 'интеграл', 'матрица', 'уравнение'
                        ])
                        
                        if is_math:
                            math_pairs.append(pair)
                        else:
                            other_pairs.append(pair)
                except:
                    continue
    
    print(f"\nЗагружено:")
    print(f"  Математических: {len(math_pairs)}")
    print(f"  Остальных: {len(other_pairs)}")
    
    # Смешиваем с нужной пропорцией
    random.seed(42)
    random.shuffle(math_pairs)
    random.shuffle(other_pairs)
    
    # Вычисляем сколько нужно каждого типа
    if max_total:
        n_math = min(int(max_total * math_ratio), len(math_pairs))
        n_other = min(max_total - n_math, len(other_pairs))
    else:
        n_math = len(math_pairs)
        n_other = int(len(math_pairs) * (1 - math_ratio) / math_ratio) if math_pairs else len(other_pairs)
        n_other = min(n_other, len(other_pairs))
    
    combined = math_pairs[:n_math] + other_pairs[:n_other]
    random.shuffle(combined)
    
    print(f"\nИтоговый датасет:")
    print(f"  Математических: {n_math}")
    print(f"  Остальных: {n_other}")
    print(f"  Всего: {len(combined)}")
    
    # Сохраняем
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in combined:
            json.dump(pair, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\nСохранено в {output_file}")
    return len(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Объединение датасетов")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="Входные файлы для объединения"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Выходной файл"
    )
    parser.add_argument(
        "--math_ratio",
        type=float,
        default=0.3,
        help="Доля математических текстов (0-1)"
    )
    parser.add_argument(
        "--max_total",
        type=int,
        default=None,
        help="Максимальное количество примеров"
    )
    
    args = parser.parse_args()
    
    combine_datasets(
        args.input_files,
        args.output_file,
        args.math_ratio,
        args.max_total
    )

