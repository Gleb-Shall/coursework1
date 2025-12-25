"""
Скрипт для оценки качества модели машинного перевода.
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
from utils.metrics import (
    calculate_bleu,
    calculate_chrf,
    formula_preservation_score,
    evaluate_translation_quality
)


def load_model(model_dir: str):
    """Загружает обученную модель."""
    print(f"Загрузка модели из {model_dir}...")
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Модель загружена на {device}")
    return model, tokenizer, device


def translate_batch(
    model,
    tokenizer,
    texts: List[str],
    device,
    max_length: int = 512,
    batch_size: int = 8
) -> List[str]:
    """Переводит батч текстов."""
    translations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Токенизация
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Генерация
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Декодирование
        batch_translations = tokenizer.batch_decode(
            translated,
            skip_special_tokens=True
        )
        
        translations.extend(batch_translations)
    
    return translations


def evaluate_model(
    model_dir: str,
    test_file: str,
    output_file: str = None,
    batch_size: int = 8,
    max_length: int = 512
):
    """Оценивает модель на тестовом наборе."""
    
    # Загрузка модели
    model, tokenizer, device = load_model(model_dir)
    
    # Загрузка тестовых данных
    print(f"Загрузка тестовых данных из {test_file}...")
    sources = []
    references = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pair = json.loads(line.strip())
            sources.append(pair["source"])
            references.append(pair["target"])
    
    print(f"Загружено {len(sources)} примеров")
    
    # Перевод
    print("Выполнение переводов...")
    predictions = translate_batch(
        model, tokenizer, sources, device,
        max_length=max_length, batch_size=batch_size
    )
    
    # Вычисление метрик
    print("Вычисление метрик...")
    
    # Стандартные метрики
    bleu_score = calculate_bleu(references, predictions)
    chrf_score = calculate_chrf(references, predictions)
    
    # Специализированные метрики
    formula_scores = [
        formula_preservation_score(ref, pred)
        for ref, pred in zip(references, predictions)
    ]
    avg_formula_preservation = sum(formula_scores) / len(formula_scores) if formula_scores else 0.0
    
    # Результаты
    results = {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "formula_preservation": avg_formula_preservation,
        "num_examples": len(sources)
    }
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*50)
    print(f"BLEU: {bleu_score:.4f}")
    print(f"chrF: {chrf_score:.4f}")
    print(f"Formula Preservation: {avg_formula_preservation:.4f}")
    print(f"Количество примеров: {len(sources)}")
    print("="*50)
    
    # Сохранение результатов
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены в {output_file}")
    
    # Сохранение примеров переводов
    examples_file = output_file.replace('.json', '_examples.json') if output_file else "results/examples.json"
    os.makedirs(os.path.dirname(examples_file) if os.path.dirname(examples_file) else ".", exist_ok=True)
    
    # Сохраняем первые 20 примеров
    examples = []
    for i in range(min(20, len(sources))):
        examples.append({
            "source": sources[i],
            "reference": references[i],
            "prediction": predictions[i],
            "formula_preservation": formula_scores[i]
        })
    
    with open(examples_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Примеры переводов сохранены в {examples_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Оценка модели машинного перевода")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Директория с обученной моделью"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/test.jsonl",
        help="Файл с тестовыми данными"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/evaluation_results.json",
        help="Файл для сохранения результатов"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Размер батча для инференса"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Максимальная длина последовательности"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_dir=args.model_dir,
        test_file=args.test_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()

