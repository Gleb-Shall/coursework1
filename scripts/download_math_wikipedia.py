"""
Скрипт для загрузки математических текстов из Wikipedia через HuggingFace.
Использует датасет Wikipedia для поиска математических статей.
"""

import os
import json
import argparse
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.formula_handler import FormulaHandler
from utils.logger import setup_logger


def download_wikipedia_math(
    output_file: str = "data/raw/wikipedia_math.jsonl",
    max_articles: int = 1000,
    logger=None
):
    """
    Загружает математические статьи из Wikipedia.
    """
    if logger is None:
        logger = setup_logger("download_wikipedia_math")
    
    try:
        from datasets import load_dataset
        
        logger.info("Загрузка Wikipedia датасета...")
        
        # Загружаем английскую Wikipedia
        dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        
        handler = FormulaHandler()
        math_articles = []
        
        logger.info(f"Поиск математических статей (максимум {max_articles})...")
        
        for article in tqdm(dataset, desc="Поиск"):
            if len(math_articles) >= max_articles:
                break
            
            text = article.get("text", "")
            title = article.get("title", "")
            
            # Проверяем, является ли статья математической
            is_math = (
                handler.has_formula(text) or 
                handler.has_math_keywords(text) or
                any(kw in title.lower() for kw in [
                    'theorem', 'proof', 'mathematics', 'algebra', 'calculus',
                    'geometry', 'analysis', 'function', 'equation', 'теорема',
                    'математика', 'алгебра', 'геометрия', 'функция'
                ])
            )
            
            if is_math:
                # Разбиваем на предложения (упрощённо)
                sentences = text.split('. ')
                for sent in sentences[:10]:  # Берём первые 10 предложений
                    if len(sent.split()) > 5 and len(sent.split()) < 100:
                        math_articles.append({
                            "source": sent.strip(),
                            "title": title,
                            "has_formula": handler.has_formula(sent)
                        })
        
        logger.info(f"Найдено {len(math_articles)} математических фрагментов")
        
        # Сохраняем
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in math_articles:
                json.dump({
                    "source": article["source"],
                    "target": ""  # Нужен перевод
                }, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Сохранено в {output_file}")
        logger.warning("Внимание: эти тексты только на английском, нужен перевод!")
        
        return output_file
    
    except ImportError:
        logger.error("Библиотека 'datasets' не установлена")
        raise
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Загрузка математических текстов из Wikipedia")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/raw/wikipedia_math.jsonl",
        help="Выходной файл"
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=1000,
        help="Максимальное количество статей для обработки"
    )
    
    args = parser.parse_args()
    
    logger = setup_logger("download_wikipedia_math")
    
    try:
        download_wikipedia_math(args.output_file, args.max_articles, logger)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

