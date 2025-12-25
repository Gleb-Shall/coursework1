"""
Базовые тесты для проверки работоспособности основных компонентов.
"""

import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_formula_handler():
    """Тест обработчика формул."""
    print("Тест 1: FormulaHandler...")
    from utils.formula_handler import FormulaHandler
    
    handler = FormulaHandler()
    text = "Let $f(x) = x^2$ be a function."
    
    assert handler.has_formula(text), "Должна быть найдена формула"
    assert handler.has_math_keywords(text), "Должны быть найдены математические ключевые слова"
    
    formulas = handler.find_formulas(text)
    assert len(formulas) > 0, "Должна быть найдена хотя бы одна формула"
    
    print("   ✓ FormulaHandler работает корректно")


def test_metrics():
    """Тест метрик."""
    print("Тест 2: Metrics...")
    from utils.metrics import calculate_bleu, formula_preservation_score
    
    references = ["This is a test."]
    predictions = ["This is a test."]
    
    bleu = calculate_bleu(references, predictions)
    assert bleu > 0, "BLEU должен быть положительным"
    
    formula_score = formula_preservation_score(
        "Let $f(x)$ be a function.",
        "Let $f(x)$ be a function."
    )
    assert formula_score == 1.0, "Формулы должны совпадать"
    
    print("   ✓ Metrics работают корректно")


def test_validation():
    """Тест валидации."""
    print("Тест 3: Validation...")
    from utils.validation import check_dependencies, check_gpu
    
    all_installed, missing = check_dependencies()
    if not all_installed:
        print(f"   ⚠ Отсутствуют пакеты: {missing}")
    else:
        print("   ✓ Все зависимости установлены")
    
    gpu_info = check_gpu()
    if gpu_info["available"]:
        print(f"   ✓ GPU доступен: {gpu_info['device_name']}")
    else:
        print("   ⚠ GPU не доступен (будет использоваться CPU)")
    
    print("   ✓ Validation работает корректно")


def test_logger():
    """Тест логирования."""
    print("Тест 4: Logger...")
    from utils.logger import setup_logger
    
    logger = setup_logger("test_logger", log_to_file=False)
    logger.info("Тестовое сообщение")
    
    print("   ✓ Logger работает корректно")


def main():
    """Запускает все тесты."""
    print("="*60)
    print("БАЗОВЫЕ ТЕСТЫ ПРОЕКТА")
    print("="*60)
    
    tests = [
        test_formula_handler,
        test_metrics,
        test_validation,
        test_logger
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ✗ Ошибка: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Результаты: {passed} пройдено, {failed} провалено")
    print("="*60)
    
    if failed == 0:
        print("✓ Все тесты пройдены успешно!")
        return 0
    else:
        print("✗ Некоторые тесты провалены")
        return 1


if __name__ == "__main__":
    sys.exit(main())

