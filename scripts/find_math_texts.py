"""
Скрипт для поиска и извлечения математических текстов из датасета.
Улучшенная фильтрация для получения большего количества математических примеров.
"""

import os
import json
import argparse
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.formula_handler import FormulaHandler


def find_math_texts(
    input_file: str,
    output_file: str,
    min_math_score: float = 0.5
):
    """
    Находит математические тексты в датасете с улучшенной фильтрацией.
    
    Args:
        input_file: Входной файл
        output_file: Выходной файл
        min_math_score: Минимальный "математический" score (0-1)
    """
    handler = FormulaHandler()
    
    print(f"Поиск математических текстов в {input_file}...")
    
    math_pairs = []
    total = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Анализ"):
            total += 1
            try:
                data = json.loads(line.strip())
                source = data.get("source", "").strip()
                target = data.get("target", "").strip()
                
                if not source or not target:
                    continue
                
                # Вычисляем "математический score"
                score = 0.0
                
                # Формулы - высокий приоритет (обязательно для математики)
                if handler.has_formula(source):
                    score += 0.6
                else:
                    # Без формул - снижаем приоритет
                    score -= 0.2
                
                # Строгие математические ключевые слова
                import re
                strict_math_keywords = [
                    r'\btheorem\b', r'\bproof\b', r'\blemma\b', r'\bcorollary\b',
                    r'\bderivative\b', r'\bintegral\b', r'\bmatrix\b', r'\bvector\b',
                    r'\bequation\b', r'\binequality\b', r'\bpolynomial\b',
                    r'\bтеорема\b', r'\bдоказательство\b', r'\bпроизводная\b',
                    r'\bинтеграл\b', r'\bматрица\b', r'\bуравнение\b'
                ]
                
                strict_matches = sum(1 for kw in strict_math_keywords if re.search(kw, source, re.IGNORECASE))
                if strict_matches > 0:
                    score += 0.4
                
                # Математические паттерны
                math_patterns = [
                    r'\$[^$]+\$',  # LaTeX формулы
                    r'\\[a-zA-Z]+\s*\{',  # LaTeX команды с аргументами
                    r'\b(sin|cos|tan|log|ln|exp|sqrt|sum|int|lim)\s*\(',  # Математические функции
                    r'\b\w+\s*=\s*\w+\s*[+\-*/]',  # Уравнения с операциями
                    r'\b\d+\s*[+\-*/=]\s*\d+',  # Арифметические выражения
                ]
                
                pattern_matches = sum(1 for p in math_patterns if re.search(p, source, re.IGNORECASE))
                score += min(pattern_matches * 0.15, 0.3)
                
                # Штраф за художественные тексты
                fiction_keywords = ['said', 'thought', 'looked', 'felt', 'went', 'came',
                                  'сказал', 'думал', 'посмотрел', 'почувствовал', 'пошёл']
                fiction_matches = sum(1 for kw in fiction_keywords if kw.lower() in source.lower())
                if fiction_matches > 2:
                    score -= 0.3
                
                # Проверяем длину (слишком короткие пропускаем)
                if len(source.split()) < 3:
                    continue
                
                # Дополнительная проверка: должны быть формулы ИЛИ строгие математические термины
                has_strict_math = (
                    handler.has_formula(source) or
                    any(kw in source.lower() for kw in [
                        'theorem', 'proof', 'lemma', 'derivative', 'integral',
                        'matrix', 'equation', 'function', 'теорема', 'доказательство',
                        'производная', 'интеграл', 'матрица', 'уравнение', 'функция'
                    ])
                )
                
                # Если score достаточно высокий И есть строгая математика, сохраняем
                if score >= min_math_score and has_strict_math:
                    math_pairs.append({
                        "source": source,
                        "target": target,
                        "math_score": score
                    })
            
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Ошибка при обработке строки: {e}")
                continue
    
    print(f"\nНайдено математических текстов: {len(math_pairs)} из {total}")
    print(f"Процент: {100 * len(math_pairs) / total:.2f}%")
    
    # Сортируем по score (лучшие первыми)
    math_pairs.sort(key=lambda x: x["math_score"], reverse=True)
    
    # Сохраняем
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in math_pairs:
            # Сохраняем без score для совместимости
            json.dump({
                "source": pair["source"],
                "target": pair["target"]
            }, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Сохранено в {output_file}")
    
    # Показываем примеры
    print("\nПримеры найденных математических текстов:")
    for i, pair in enumerate(math_pairs[:5], 1):
        print(f"\n{i}. Score: {pair['math_score']:.2f}")
        print(f"   EN: {pair['source'][:100]}...")
        print(f"   RU: {pair['target'][:100]}...")
    
    return len(math_pairs)


def main():
    parser = argparse.ArgumentParser(description="Поиск математических текстов в датасете")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/raw/wikimatrix_en_ru.jsonl",
        help="Входной файл"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/raw/math_texts.jsonl",
        help="Выходной файл с математическими текстами"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.3,
        help="Минимальный математический score (0-1, чем выше, тем строже фильтрация)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Ошибка: файл {args.input_file} не найден")
        return
    
    find_math_texts(args.input_file, args.output_file, args.min_score)


if __name__ == "__main__":
    main()

