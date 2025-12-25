"""
Utilities for working with LaTeX mathematical formulas.
Formula processing for preprocessing texts in machine translation.
"""

import re
from typing import List, Tuple, Dict


class FormulaHandler:
    """Class for handling LaTeX formulas in texts."""
    
    def __init__(self, preserve_formulas: bool = True):
        """
        Args:
            preserve_formulas: If True, formulas are replaced with tokens,
                              otherwise remain as is
        """
        self.preserve_formulas = preserve_formulas
        # Patterns for finding formulas
        self.inline_pattern = re.compile(r'\$([^$]+)\$')
        self.display_pattern = re.compile(r'\$\$([^$]+)\$\$')
        self.latex_env_pattern = re.compile(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', re.DOTALL)
        
    def find_formulas(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Находит все формулы в тексте.
        
        Returns:
            Список кортежей (formula, start_pos, end_pos)
        """
        formulas = []
        
        # Inline формулы $...$
        for match in self.inline_pattern.finditer(text):
            formulas.append((match.group(1), match.start(), match.end()))
        
        # Display формулы $$...$$
        for match in self.display_pattern.finditer(text):
            formulas.append((match.group(1), match.start(), match.end()))
        
        # LaTeX окружения
        for match in self.latex_env_pattern.finditer(text):
            formulas.append((match.group(0), match.start(), match.end()))
        
        return formulas
    
    def has_formula(self, text: str) -> bool:
        """Проверяет, содержит ли текст формулы."""
        return bool(self.inline_pattern.search(text) or 
                   self.display_pattern.search(text) or
                   self.latex_env_pattern.search(text))
    
    def has_math_keywords(self, text: str) -> bool:
        """
        Проверяет наличие математических ключевых слов.
        Используется для фильтрации математических текстов.
        Строгая фильтрация для выделения математических текстов из общего корпуса.
        """
        # Строгие математические термины (высокий приоритет)
        strict_math_keywords_en = [
            # Теоремы и доказательства
            'theorem', 'lemma', 'corollary', 'proposition', 'conjecture',
            'proof', 'prove', 'demonstrate',
            # Функции и отображения
            'function', 'mapping', 'domain', 'codomain', 'range',
            'continuous', 'differentiable', 'derivative', 'integral',
            # Уравнения и неравенства
            'equation', 'inequality', 'solve', 'solution',
            # Линейная алгебра
            'matrix', 'vector', 'space', 'dimension', 'basis',
            # Анализ
            'limit', 'series', 'sequence', 'converge', 'diverge',
            # Множества и логика
            'set', 'subset', 'element', 'exists', 'forall', 'implies',
        ]
        
        strict_math_keywords_ru = [
            # Теоремы и доказательства
            'теорема', 'лемма', 'следствие', 'предложение',
            'доказательство', 'доказать',
            # Функции
            'функция', 'отображение', 'область', 'непрерывная',
            'дифференцируемая', 'производная', 'интеграл',
            # Уравнения
            'уравнение', 'неравенство', 'решить', 'решение',
            # Линейная алгебра
            'матрица', 'вектор', 'пространство', 'размерность', 'базис',
            # Анализ
            'предел', 'ряд', 'последовательность', 'сходится',
            # Множества
            'множество', 'подмножество', 'элемент', 'существует', 'для всех',
        ]
        
        text_lower = text.lower()
        all_keywords = strict_math_keywords_en + strict_math_keywords_ru
        
        # Проверяем наличие строгих математических терминов
        matches = [kw for kw in all_keywords if kw in text_lower]
        
        # Если есть хотя бы один строгий термин - это математика
        if len(matches) > 0:
            return True
        
        return False
    
    def has_math_patterns(self, text: str) -> bool:
        """
        Проверяет наличие математических паттернов в тексте.
        """
        # LaTeX формулы
        if re.search(r'\$[^$]+\$', text):
            return True
        
        # LaTeX команды
        latex_commands = [
            r'\\sum', r'\\int', r'\\lim', r'\\forall', r'\\exists',
            r'\\in', r'\\subset', r'\\cup', r'\\cap', r'\\setminus',
            r'\\frac', r'\\sqrt', r'\\sin', r'\\cos', r'\\tan',
            r'\\log', r'\\ln', r'\\exp', r'\\det', r'\\dim'
        ]
        
        for cmd in latex_commands:
            if re.search(cmd, text):
                return True
        
        # Математические операторы и символы
        math_operators = [
            r'\s*=\s*',  # Равенство
            r'\s*<\s*|\s*>\s*',  # Неравенства
            r'\s*≤\s*|\s*≥\s*',  # Неравенства (Unicode)
            r'\s*≠\s*',  # Неравенство
            r'\s*∈\s*',  # Принадлежность
            r'\s*⊂\s*|\s*⊆\s*',  # Подмножество
        ]
        
        operator_count = sum(1 for op in math_operators if re.search(op, text))
        if operator_count >= 2:  # Если есть несколько операторов
            return True
        
        # Математические выражения с переменными
        math_expressions = [
            r'\b\w+\s*=\s*\w+[+\-*/]',  # Уравнения с операциями
            r'\b\w+\([^)]+\)\s*=',  # Функции с равенством
            r'\b\d+[+\-*/]\d+',  # Арифметические выражения
            r'\b\w+\^?\d+',  # Степени (x^2, f^3)
        ]
        
        for pattern in math_expressions:
            if re.search(pattern, text):
                return True
        
        return False
    
    def replace_formulas_with_tokens(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Заменяет формулы на токены и возвращает маппинг.
        
        Returns:
            (processed_text, formula_mapping)
        """
        formulas = self.find_formulas(text)
        formula_mapping = {}
        processed_text = text
        offset = 0
        
        for i, (formula, start, end) in enumerate(formulas):
            token = f"<FORMULA_{i}>"
            formula_mapping[token] = formula
            # Заменяем с учётом смещения
            processed_text = (
                processed_text[:start + offset] + 
                token + 
                processed_text[end + offset:]
            )
            offset += len(token) - (end - start)
        
        return processed_text, formula_mapping
    
    def restore_formulas(self, text: str, formula_mapping: Dict[str, str]) -> str:
        """Восстанавливает формулы из токенов."""
        restored_text = text
        for token, formula in formula_mapping.items():
            restored_text = restored_text.replace(token, f"${formula}$")
        return restored_text
    
    def is_valid_formula(self, formula: str) -> bool:
        """
        Проверяет валидность формулы (базовая проверка).
        """
        # Проверка на незакрытые скобки
        if formula.count('{') != formula.count('}'):
            return False
        if formula.count('(') != formula.count(')'):
            return False
        if formula.count('[') != formula.count(']'):
            return False
        
        # Проверка на битые команды
        if re.search(r'\\[a-zA-Z]+\s*$', formula):
            return False
        
        return True
    
    def clean_formula(self, formula: str) -> str:
        """Очищает формулу от лишних пробелов."""
        # Убираем лишние пробелы, но сохраняем структуру
        formula = re.sub(r'\s+', ' ', formula)
        formula = formula.strip()
        return formula


def filter_math_texts(texts: List[str], handler: FormulaHandler) -> List[bool]:
    """
    Фильтрует тексты, содержащие математику.
    Использует комбинацию проверок: формулы, ключевые слова, паттерны.
    
    Returns:
        Список булевых значений (True если текст математический)
    """
    return [
        handler.has_formula(text) or 
        handler.has_math_keywords(text) or 
        handler.has_math_patterns(text)
        for text in texts
    ]

