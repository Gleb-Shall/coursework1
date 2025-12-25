"""
Скрипт для быстрого перевода текста с использованием обученной модели.
"""

import argparse
import torch
from transformers import MarianMTModel, MarianTokenizer


def translate_text(
    model_dir: str,
    text: str,
    max_length: int = 512
) -> str:
    """Переводит текст с помощью модели."""
    # Загрузка модели
    print(f"Загрузка модели из {model_dir}...")
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Токенизация
    inputs = tokenizer(
        text,
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
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translation


def main():
    parser = argparse.ArgumentParser(description="Перевод текста")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Директория с обученной моделью"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Текст для перевода"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Максимальная длина"
    )
    
    args = parser.parse_args()
    
    translation = translate_text(args.model_dir, args.text, args.max_length)
    
    print("\n" + "="*60)
    print("ИСХОДНЫЙ ТЕКСТ:")
    print("="*60)
    print(args.text)
    print("\n" + "="*60)
    print("ПЕРЕВОД:")
    print("="*60)
    print(translation)
    print("="*60)


if __name__ == "__main__":
    main()

