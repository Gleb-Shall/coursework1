"""
Скрипт для сравнения нескольких моделей на одном тестовом наборе.
"""

import os
import json
import argparse
from typing import List, Dict

import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import calculate_bleu, calculate_chrf, formula_preservation_score


def load_model(model_dir: str):
    """Загружает модель."""
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device


def translate_batch(model, tokenizer, texts: List[str], device, max_length: int = 512):
    """Переводит батч текстов."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translations


def evaluate_model(model_dir: str, test_file: str, max_length: int = 512):
    """Оценивает модель."""
    print(f"Оценка модели: {model_dir}")
    
    model, tokenizer, device = load_model(model_dir)
    
    sources = []
    references = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pair = json.loads(line.strip())
            sources.append(pair["source"])
            references.append(pair["target"])
    
    predictions = []
    batch_size = 8
    
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        batch_preds = translate_batch(model, tokenizer, batch, device, max_length)
        predictions.extend(batch_preds)
    
    # Метрики
    bleu = calculate_bleu(references, predictions)
    chrf = calculate_chrf(references, predictions)
    
    formula_scores = [
        formula_preservation_score(ref, pred)
        for ref, pred in zip(references, predictions)
    ]
    avg_formula = sum(formula_scores) / len(formula_scores) if formula_scores else 0.0
    
    return {
        "model_dir": model_dir,
        "bleu": bleu,
        "chrf": chrf,
        "formula_preservation": avg_formula,
        "num_examples": len(sources)
    }


def main():
    parser = argparse.ArgumentParser(description="Сравнение моделей")
    parser.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Директории с моделями для сравнения"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/test.jsonl",
        help="Тестовый файл"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/model_comparison.json",
        help="Файл для сохранения результатов"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Максимальная длина"
    )
    
    args = parser.parse_args()
    
    results = []
    
    for model_dir in args.model_dirs:
        if not os.path.exists(model_dir):
            print(f"Предупреждение: модель {model_dir} не найдена, пропускаем")
            continue
        
        try:
            result = evaluate_model(model_dir, args.test_file, args.max_length)
            results.append(result)
        except Exception as e:
            print(f"Ошибка при оценке {model_dir}: {e}")
            continue
    
    # Вывод результатов
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МОДЕЛЕЙ")
    print("="*80)
    print(f"{'Модель':<40} {'BLEU':<10} {'chrF':<10} {'Formula':<10}")
    print("-"*80)
    
    for result in results:
        model_name = os.path.basename(result["model_dir"])
        print(f"{model_name:<40} {result['bleu']:<10.4f} {result['chrf']:<10.4f} {result['formula_preservation']:<10.4f}")
    
    print("="*80)
    
    # Сохранение
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в {args.output_file}")


if __name__ == "__main__":
    main()

