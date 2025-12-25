"""
Дополнительные метрики для оценки качества перевода математических текстов.
"""

import re
from typing import List, Dict
from sacrebleu import BLEU, CHRF
import numpy as np


def calculate_bleu(references: List[str], predictions: List[str]) -> float:
    """Вычисляет BLEU score."""
    bleu = BLEU()
    score = bleu.corpus_score(predictions, [references])
    return score.score


def calculate_chrf(references: List[str], predictions: List[str]) -> float:
    """Вычисляет chrF score."""
    chrf = CHRF()
    score = chrf.corpus_score(predictions, [references])
    return score.score


def extract_formulas(text: str) -> List[str]:
    """Извлекает все формулы из текста."""
    inline_pattern = re.compile(r'\$([^$]+)\$')
    display_pattern = re.compile(r'\$\$([^$]+)\$\$')
    
    formulas = []
    for match in inline_pattern.finditer(text):
        formulas.append(match.group(1))
    for match in display_pattern.finditer(text):
        formulas.append(match.group(1))
    
    return formulas


def formula_preservation_score(reference: str, prediction: str) -> float:
    """
    Оценивает сохранение формул в переводе.
    Возвращает долю формул из reference, которые присутствуют в prediction.
    """
    ref_formulas = set(extract_formulas(reference))
    pred_formulas = set(extract_formulas(prediction))
    
    if not ref_formulas:
        return 1.0  # Нет формул - считаем идеально
    
    if not pred_formulas:
        return 0.0  # Формулы потеряны
    
    # Сравниваем формулы (нормализованные)
    ref_normalized = {f.strip().lower() for f in ref_formulas}
    pred_normalized = {f.strip().lower() for f in pred_formulas}
    
    intersection = ref_normalized & pred_normalized
    return len(intersection) / len(ref_normalized) if ref_normalized else 0.0


def math_term_accuracy(reference: str, prediction: str, 
                      math_terms: Dict[str, List[str]]) -> float:
    """
    Оценивает точность перевода математических терминов.
    
    Args:
        reference: Референсный перевод
        prediction: Предсказанный перевод
        math_terms: Словарь {en_term: [ru_terms]}
    """
    if not math_terms:
        return 1.0
    
    ref_lower = reference.lower()
    pred_lower = prediction.lower()
    
    correct = 0
    total = 0
    
    for en_term, ru_terms in math_terms.items():
        if en_term.lower() in ref_lower:
            total += 1
            if any(ru_term.lower() in pred_lower for ru_term in ru_terms):
                correct += 1
    
    return correct / total if total > 0 else 1.0


def evaluate_translation_quality(
    references: List[str],
    predictions: List[str],
    math_terms: Dict[str, List[str]] = None
) -> Dict[str, float]:
    """
    Комплексная оценка качества перевода.
    
    Returns:
        Словарь с метриками
    """
    metrics = {}
    
    # Стандартные метрики
    metrics['bleu'] = calculate_bleu(references, predictions)
    metrics['chrf'] = calculate_chrf(references, predictions)
    
    # Специализированные метрики
    formula_scores = [
        formula_preservation_score(ref, pred)
        for ref, pred in zip(references, predictions)
    ]
    metrics['formula_preservation'] = np.mean(formula_scores)
    
    if math_terms:
        term_scores = [
            math_term_accuracy(ref, pred, math_terms)
            for ref, pred in zip(references, predictions)
        ]
        metrics['math_term_accuracy'] = np.mean(term_scores)
    
    return metrics

