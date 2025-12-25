"""
Пример использования утилит проекта.
"""

from utils.formula_handler import FormulaHandler
from utils.metrics import calculate_bleu, formula_preservation_score

# Пример работы с формулами
print("=" * 60)
print("Пример 1: Работа с формулами")
print("=" * 60)

handler = FormulaHandler()

text = "Let $f(x) = x^2$ be a function. Then $\\int_0^1 f(x) dx = \\frac{1}{3}$."

print(f"Исходный текст: {text}")
print(f"Содержит формулы: {handler.has_formula(text)}")
print(f"Содержит математические ключевые слова: {handler.has_math_keywords(text)}")

formulas = handler.find_formulas(text)
print(f"Найдено формул: {len(formulas)}")
for i, (formula, start, end) in enumerate(formulas):
    print(f"  Формула {i+1}: {formula}")

# Пример фильтрации
print("\n" + "=" * 60)
print("Пример 2: Фильтрация математических текстов")
print("=" * 60)

texts = [
    "This is a regular sentence.",
    "Let $f(x)$ be a continuous function.",
    "The theorem states that for all $x$, we have $x^2 \\geq 0$.",
    "Hello world!"
]

for text in texts:
    is_math = handler.has_formula(text) or handler.has_math_keywords(text)
    print(f"{'✓' if is_math else '✗'} {text}")

# Пример оценки метрик
print("\n" + "=" * 60)
print("Пример 3: Оценка метрик")
print("=" * 60)

reference = "Пусть $f(x)$ — непрерывная функция."
prediction = "Let $f(x)$ be a continuous function."

# Для BLEU нужны списки
references = [reference]
predictions = [prediction]

# Formula preservation (для обратного перевода)
formula_score = formula_preservation_score(reference, prediction)
print(f"Formula Preservation Score: {formula_score:.4f}")

print("\n" + "=" * 60)
print("Примеры готовы!")
print("=" * 60)

