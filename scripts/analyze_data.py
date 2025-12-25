"""
Скрипт для анализа датасета перед обучением.
"""

import os
import json
import argparse
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.formula_handler import FormulaHandler


def load_data(file_path: str) -> List[Dict]:
    """Загружает данные из JSONL файла."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def analyze_dataset(data_file: str, output_dir: str = "results/analysis"):
    """Анализирует датасет и создаёт отчёт."""
    print(f"Анализ датасета: {data_file}")
    
    data = load_data(data_file)
    print(f"Загружено {len(data)} примеров")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Статистика по длине
    source_lengths = [len(pair["source"].split()) for pair in data]
    target_lengths = [len(pair["target"].split()) for pair in data]
    
    stats = {
        "total_examples": len(data),
        "source_length": {
            "mean": np.mean(source_lengths),
            "median": np.median(source_lengths),
            "min": np.min(source_lengths),
            "max": np.max(source_lengths),
            "std": np.std(source_lengths)
        },
        "target_length": {
            "mean": np.mean(target_lengths),
            "median": np.median(target_lengths),
            "min": np.min(target_lengths),
            "max": np.max(target_lengths),
            "std": np.std(target_lengths)
        }
    }
    
    # Анализ формул
    formula_handler = FormulaHandler()
    examples_with_formulas = 0
    examples_with_math_keywords = 0
    total_formulas = 0
    
    for pair in data:
        source = pair["source"]
        if formula_handler.has_formula(source):
            examples_with_formulas += 1
            total_formulas += len(formula_handler.find_formulas(source))
        if formula_handler.has_math_keywords(source):
            examples_with_math_keywords += 1
    
    stats["formulas"] = {
        "examples_with_formulas": examples_with_formulas,
        "examples_with_math_keywords": examples_with_math_keywords,
        "total_formulas": total_formulas,
        "avg_formulas_per_example": total_formulas / len(data) if data else 0
    }
    
    # Сохранение статистики
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("="*60)
    print(f"Всего примеров: {stats['total_examples']}")
    print(f"\nДлина исходных текстов:")
    print(f"  Средняя: {stats['source_length']['mean']:.2f}")
    print(f"  Медиана: {stats['source_length']['median']:.2f}")
    print(f"  Мин/Макс: {stats['source_length']['min']}/{stats['source_length']['max']}")
    print(f"\nДлина целевых текстов:")
    print(f"  Средняя: {stats['target_length']['mean']:.2f}")
    print(f"  Медиана: {stats['target_length']['median']:.2f}")
    print(f"  Мин/Макс: {stats['target_length']['min']}/{stats['target_length']['max']}")
    print(f"\nФормулы:")
    print(f"  Примеров с формулами: {stats['formulas']['examples_with_formulas']}")
    print(f"  Примеров с мат. ключевыми словами: {stats['formulas']['examples_with_math_keywords']}")
    print(f"  Всего формул: {stats['formulas']['total_formulas']}")
    print(f"  Среднее формул на пример: {stats['formulas']['avg_formulas_per_example']:.2f}")
    print("="*60)
    
    # Визуализация (если matplotlib доступен)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Распределение длин исходных текстов
        axes[0, 0].hist(source_lengths, bins=50, edgecolor='black')
        axes[0, 0].set_title('Распределение длины исходных текстов')
        axes[0, 0].set_xlabel('Количество слов')
        axes[0, 0].set_ylabel('Частота')
        
        # Распределение длин целевых текстов
        axes[0, 1].hist(target_lengths, bins=50, edgecolor='black', color='orange')
        axes[0, 1].set_title('Распределение длины целевых текстов')
        axes[0, 1].set_xlabel('Количество слов')
        axes[0, 1].set_ylabel('Частота')
        
        # Соотношение длин
        axes[1, 0].scatter(source_lengths, target_lengths, alpha=0.5, s=1)
        axes[1, 0].set_title('Соотношение длин исходных и целевых текстов')
        axes[1, 0].set_xlabel('Длина исходного текста')
        axes[1, 0].set_ylabel('Длина целевого текста')
        
        # Распределение соотношения длин
        length_ratios = [t/s if s > 0 else 0 for s, t in zip(source_lengths, target_lengths)]
        axes[1, 1].hist(length_ratios, bins=50, edgecolor='black', color='green')
        axes[1, 1].set_title('Распределение соотношения длин (target/source)')
        axes[1, 1].set_xlabel('Соотношение')
        axes[1, 1].set_ylabel('Частота')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, "dataset_analysis.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nГрафики сохранены в {plot_file}")
        plt.close()
    except ImportError:
        print("\nMatplotlib не установлен, пропускаем визуализацию")
    
    # Примеры
    examples_file = os.path.join(output_dir, "examples.json")
    examples = []
    for i, pair in enumerate(data[:10]):  # Первые 10 примеров
        examples.append({
            "index": i,
            "source": pair["source"],
            "target": pair["target"],
            "has_formula": formula_handler.has_formula(pair["source"]),
            "has_math_keywords": formula_handler.has_math_keywords(pair["source"])
        })
    
    with open(examples_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Примеры сохранены в {examples_file}")
    print(f"\nПолная статистика сохранена в {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Анализ датасета")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Путь к файлу с данными (JSONL)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis",
        help="Директория для сохранения результатов анализа"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Ошибка: файл {args.data_file} не найден")
        return
    
    analyze_dataset(args.data_file, args.output_dir)


if __name__ == "__main__":
    main()

