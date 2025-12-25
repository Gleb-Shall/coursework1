"""
Скрипт для обучения модели машинного перевода.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TranslationDataset(Dataset):
    """Датасет для машинного перевода."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Загрузка данных из {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line.strip())
                self.data.append(pair)
        
        print(f"Загружено {len(self.data)} примеров")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pair = self.data[idx]
        source = pair["source"]
        target = pair["target"]
        
        # Токенизация
        source_encodings = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encodings = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encodings['input_ids'].squeeze(),
            'attention_mask': source_encodings['attention_mask'].squeeze(),
            'labels': target_encodings['input_ids'].squeeze()
        }


def load_model_and_tokenizer(model_name: str = "Helsinki-NLP/opus-mt-en-ru"):
    """Загружает модель и токенизатор."""
    print(f"Загрузка модели {model_name}...")
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    print("Модель загружена")
    return model, tokenizer


def create_compute_metrics(tokenizer):
    """Создаёт функцию compute_metrics с токенизатором."""
    def compute_metrics(eval_pred):
        """Вычисляет метрики для оценки."""
        predictions, labels = eval_pred
        
        # Декодируем предсказания
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Заменяем -100 на pad_token_id для декодирования
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Вычисляем BLEU
        from sacrebleu import BLEU
        bleu = BLEU()
        bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels])
        
        return {"bleu": bleu_score.score}
    
    return compute_metrics


def train_model(
    train_file: str,
    val_file: str,
    model_name: str = "Helsinki-NLP/opus-mt-en-ru",
    output_dir: str = "models/checkpoint",
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    warmup_steps: int = 500,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    fp16: bool = True,
    save_steps: int = 1000,
    eval_steps: int = 500,
    logging_steps: int = 100
):
    """Обучает модель."""
    
    # Загрузка модели
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Загрузка данных
    train_dataset = TranslationDataset(train_file, tokenizer, max_length)
    val_dataset = TranslationDataset(val_file, tokenizer, max_length)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=fp16,
        dataloader_num_workers=4,
        report_to="none",  # Отключаем wandb/tensorboard по умолчанию
        save_total_limit=3,
        push_to_hub=False
    )
    
    # Функция для вычисления метрик
    compute_metrics_fn = create_compute_metrics(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn
    )
    
    # Обучение
    print("Начало обучения...")
    trainer.train()
    
    # Сохранение финальной модели
    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"Модель сохранена в {final_model_dir}")
    
    # Финальная оценка
    print("Финальная оценка на валидационном наборе...")
    eval_results = trainer.evaluate()
    print(f"Результаты оценки: {eval_results}")
    
    return trainer, eval_results


def main():
    parser = argparse.ArgumentParser(description="Обучение модели машинного перевода")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train.jsonl",
        help="Файл с обучающими данными"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/processed/val.jsonl",
        help="Файл с валидационными данными"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-ru",
        help="Название pretrained модели"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoint",
        help="Директория для сохранения модели"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Размер батча"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Количество эпох"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Количество warmup шагов"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Максимальная длина последовательности"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Шаги накопления градиента"
    )
    parser.add_argument(
        "--no_fp16",
        dest="fp16",
        action="store_false",
        help="Отключить mixed precision"
    )
    
    args = parser.parse_args()
    
    # Проверка наличия GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    if device.type == "cpu":
        print("Предупреждение: GPU не обнаружен, обучение может быть медленным")
    
    train_model(
        train_file=args.train_file,
        val_file=args.val_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16
    )


if __name__ == "__main__":
    main()

