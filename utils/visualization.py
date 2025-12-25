"""
Утилиты для визуализации результатов обучения и оценки.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


def plot_training_history(log_dir: str, output_file: str = "results/training_history.png"):
    """
    Строит графики истории обучения из логов.
    
    Args:
        log_dir: Директория с логами обучения
        output_file: Файл для сохранения графика
    """
    # Ищем файлы с метриками
    train_losses = []
    eval_losses = []
    eval_bleu = []
    steps = []
    
    # Пытаемся найти trainer_state.json
    trainer_state_file = os.path.join(log_dir, "trainer_state.json")
    if os.path.exists(trainer_state_file):
        with open(trainer_state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
            if "log_history" in state:
                for entry in state["log_history"]:
                    if "step" in entry:
                        steps.append(entry["step"])
                    if "loss" in entry:
                        train_losses.append(entry["loss"])
                    if "eval_loss" in entry:
                        eval_losses.append(entry["eval_loss"])
                    if "eval_bleu" in entry:
                        eval_bleu.append(entry["eval_bleu"])
    
    if not steps:
        print("Не найдены данные для визуализации")
        return
    
    # Создаём графики
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Loss
    if train_losses:
        axes[0].plot(steps[:len(train_losses)], train_losses, label='Train Loss', color='blue')
    if eval_losses:
        eval_steps = [s for s, e in zip(steps, eval_losses) if e is not None]
        eval_losses_clean = [e for e in eval_losses if e is not None]
        axes[0].plot(eval_steps, eval_losses_clean, label='Eval Loss', color='red', marker='o')
    
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # BLEU
    if eval_bleu:
        eval_bleu_steps = [s for s, b in zip(steps, eval_bleu) if b is not None]
        eval_bleu_clean = [b for b in eval_bleu if b is not None]
        axes[1].plot(eval_bleu_steps, eval_bleu_clean, label='BLEU Score', color='green', marker='s')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('BLEU')
        axes[1].set_title('Validation BLEU Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"График сохранён в {output_file}")
    plt.close()


def plot_metrics_comparison(
    results: List[Dict],
    output_file: str = "results/metrics_comparison.png"
):
    """
    Сравнивает метрики нескольких моделей.
    
    Args:
        results: Список словарей с метриками моделей
        output_file: Файл для сохранения графика
    """
    if not results:
        print("Нет данных для сравнения")
        return
    
    model_names = [os.path.basename(r.get("model_dir", f"Model {i}")) for i, r in enumerate(results)]
    bleu_scores = [r.get("bleu", 0) for r in results]
    chrf_scores = [r.get("chrf", 0) for r in results]
    formula_scores = [r.get("formula_preservation", 0) for r in results]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, bleu_scores, width, label='BLEU', alpha=0.8)
    ax.bar(x, chrf_scores, width, label='chrF', alpha=0.8)
    ax.bar(x + width, formula_scores, width, label='Formula Preservation', alpha=0.8)
    
    ax.set_xlabel('Модели')
    ax.set_ylabel('Score')
    ax.set_title('Сравнение метрик моделей')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"График сохранён в {output_file}")
    plt.close()


def plot_translation_examples(
    examples: List[Dict],
    output_file: str = "results/translation_examples.png",
    max_examples: int = 5
):
    """
    Визуализирует примеры переводов.
    
    Args:
        examples: Список примеров с source, reference, prediction
        output_file: Файл для сохранения
        max_examples: Максимальное количество примеров для отображения
    """
    if not examples:
        print("Нет примеров для визуализации")
        return
    
    examples = examples[:max_examples]
    
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 3 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    
    for i, example in enumerate(examples):
        ax = axes[i]
        ax.axis('off')
        
        source = example.get("source", "")
        reference = example.get("reference", "")
        prediction = example.get("prediction", "")
        
        text = f"Source: {source}\n\nReference: {reference}\n\nPrediction: {prediction}"
        
        ax.text(0.05, 0.5, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace', wrap=True)
        
        ax.set_title(f"Example {i+1}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Примеры сохранены в {output_file}")
    plt.close()

