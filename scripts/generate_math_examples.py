"""
Генерация математических примеров для обучения.
Создаёт базовый набор математических текстов с формулами.
"""

import json
import os

# Базовые математические примеры
MATH_EXAMPLES = [
    # Функции и производные
    {"source": "Let $f(x) = x^2$ be a function. Then $f'(x) = 2x$.", 
     "target": "Пусть $f(x) = x^2$ — функция. Тогда $f'(x) = 2x$."},
    {"source": "The derivative of $\\sin(x)$ is $\\cos(x)$.", 
     "target": "Производная от $\\sin(x)$ равна $\\cos(x)$."},
    {"source": "If $f(x) = e^x$, then $f'(x) = e^x$.", 
     "target": "Если $f(x) = e^x$, то $f'(x) = e^x$."},
    
    # Интегралы
    {"source": "The integral of $x^2$ is $\\frac{x^3}{3} + C$.", 
     "target": "Интеграл от $x^2$ равен $\\frac{x^3}{3} + C$."},
    {"source": "We have $\\int_0^1 x dx = \\frac{1}{2}$.", 
     "target": "Имеем $\\int_0^1 x dx = \\frac{1}{2}$."},
    {"source": "The definite integral $\\int_a^b f(x) dx$ represents the area under the curve.", 
     "target": "Определённый интеграл $\\int_a^b f(x) dx$ представляет площадь под кривой."},
    
    # Теоремы
    {"source": "By the mean value theorem, there exists $c \\in (a,b)$ such that $f'(c) = \\frac{f(b)-f(a)}{b-a}$.", 
     "target": "По теореме о среднем значении существует $c \\in (a,b)$ такое, что $f'(c) = \\frac{f(b)-f(a)}{b-a}$."},
    {"source": "The Pythagorean theorem states that $a^2 + b^2 = c^2$ for a right triangle.", 
     "target": "Теорема Пифагора утверждает, что $a^2 + b^2 = c^2$ для прямоугольного треугольника."},
    
    # Матрицы и векторы
    {"source": "A matrix $A$ is invertible if and only if $\\det(A) \\neq 0$.", 
     "target": "Матрица $A$ обратима тогда и только тогда, когда $\\det(A) \\neq 0$."},
    {"source": "The dot product of vectors $\\mathbf{u}$ and $\\mathbf{v}$ is $\\mathbf{u} \\cdot \\mathbf{v} = |\\mathbf{u}||\\mathbf{v}|\\cos(\\theta)$.", 
     "target": "Скалярное произведение векторов $\\mathbf{u}$ и $\\mathbf{v}$ равно $\\mathbf{u} \\cdot \\mathbf{v} = |\\mathbf{u}||\\mathbf{v}|\\cos(\\theta)$."},
    
    # Пределы
    {"source": "The limit $\\lim_{x \\to 0} \\frac{\\sin(x)}{x} = 1$ is a fundamental result.", 
     "target": "Предел $\\lim_{x \\to 0} \\frac{\\sin(x)}{x} = 1$ является фундаментальным результатом."},
    {"source": "We say that $\\lim_{n \\to \\infty} a_n = L$ if for every $\\epsilon > 0$ there exists $N$ such that $|a_n - L| < \\epsilon$ for all $n > N$.", 
     "target": "Говорим, что $\\lim_{n \\to \\infty} a_n = L$, если для любого $\\epsilon > 0$ существует $N$ такое, что $|a_n - L| < \\epsilon$ для всех $n > N$."},
    
    # Ряды
    {"source": "The geometric series $\\sum_{n=0}^{\\infty} r^n$ converges if $|r| < 1$.", 
     "target": "Геометрический ряд $\\sum_{n=0}^{\\infty} r^n$ сходится, если $|r| < 1$."},
    {"source": "The Taylor series of $e^x$ is $e^x = \\sum_{n=0}^{\\infty} \\frac{x^n}{n!}$.", 
     "target": "Ряд Тейлора для $e^x$ равен $e^x = \\sum_{n=0}^{\\infty} \\frac{x^n}{n!}$."},
    
    # Уравнения
    {"source": "The quadratic equation $ax^2 + bx + c = 0$ has solutions $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$.", 
     "target": "Квадратное уравнение $ax^2 + bx + c = 0$ имеет решения $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$."},
    {"source": "To solve the equation $2x + 3 = 7$, we subtract 3 from both sides to get $2x = 4$, so $x = 2$.", 
     "target": "Чтобы решить уравнение $2x + 3 = 7$, вычитаем 3 из обеих частей: $2x = 4$, значит $x = 2$."},
    
    # Множества
    {"source": "The set $A = \\{1, 2, 3\\}$ has three elements.", 
     "target": "Множество $A = \\{1, 2, 3\\}$ содержит три элемента."},
    {"source": "For sets $A$ and $B$, we have $A \\cup B = \\{x : x \\in A \\text{ or } x \\in B\\}$.", 
     "target": "Для множеств $A$ и $B$ имеем $A \\cup B = \\{x : x \\in A \\text{ или } x \\in B\\}$."},
    
    # Доказательства
    {"source": "We prove by contradiction. Assume that $\\sqrt{2}$ is rational.", 
     "target": "Доказываем от противного. Предположим, что $\\sqrt{2}$ рационально."},
    {"source": "The proof follows from the definition of continuity.", 
     "target": "Доказательство следует из определения непрерывности."},
    
    # Определения
    {"source": "A function $f$ is continuous at $a$ if $\\lim_{x \\to a} f(x) = f(a)$.", 
     "target": "Функция $f$ непрерывна в точке $a$, если $\\lim_{x \\to a} f(x) = f(a)$."},
    {"source": "A real number $L$ is the limit of $f(x)$ as $x$ approaches $a$ if for every $\\epsilon > 0$ there exists $\\delta > 0$ such that $|f(x) - L| < \\epsilon$ whenever $0 < |x - a| < \\delta$.", 
     "target": "Действительное число $L$ является пределом $f(x)$ при $x$, стремящемся к $a$, если для любого $\\epsilon > 0$ существует $\\delta > 0$ такое, что $|f(x) - L| < \\epsilon$ всякий раз, когда $0 < |x - a| < \\delta$."},
]

def generate_math_dataset(output_file: str, num_copies: int = 1):
    """Генерирует датасет математических примеров."""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in range(num_copies):
            for example in MATH_EXAMPLES:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
    
    print(f"Создано {len(MATH_EXAMPLES) * num_copies} математических примеров")
    print(f"Сохранено в {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="data/raw/math_generated.jsonl")
    parser.add_argument("--num_copies", type=int, default=3, help="Количество копий каждого примера")
    args = parser.parse_args()
    
    generate_math_dataset(args.output_file, args.num_copies)

