"""
Скрипт для загрузки датасета WikiMatrix EN-RU.
Поддерживает несколько источников: HuggingFace, OPUS, альтернативные датасеты.
"""

import os
import json
import sys
import argparse
import urllib.request
import gzip
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger


def download_from_huggingface(output_dir: str, max_samples: int = None, logger=None):
    """Попытка загрузки через HuggingFace datasets."""
    try:
        from datasets import load_dataset
        
        logger.info("Попытка загрузки через HuggingFace...")
        
        # Пробуем разные варианты названий WikiMatrix
        dataset_names = [
            ("facebook/wikimatrix", "en-ru"),
            ("wikimatrix", "en-ru"),
            ("opus/wikimatrix", "en-ru"),
            # Попробуем также другие варианты
            ("GEM/wiki_auto_large", None),  # Alternative if available
        ]
        
        for dataset_name, config in dataset_names:
            try:
                logger.info(f"Пробуем: {dataset_name} с конфигом {config}")
                dataset = load_dataset(dataset_name, config, split="train")
                
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                
                logger.info(f"Загружено {len(dataset)} примеров")
                return dataset
            except Exception as e:
                logger.debug(f"Не удалось загрузить {dataset_name}: {e}")
                continue
        
        raise Exception("Не удалось загрузить ни один вариант через HuggingFace")
    
    except ImportError:
        raise Exception("Библиотека 'datasets' не установлена. Установите: pip install datasets")
    except Exception as e:
        raise Exception(f"Ошибка HuggingFace: {e}")


def download_from_opus_url(output_dir: str, max_samples: int = None, logger=None):
    """
    Загрузка с OPUS через прямой URL.
    WikiMatrix доступен на https://opus.nlpl.eu/WikiMatrix/corpus/version/WikiMatrix
    """
    import tempfile
    
    logger.info("Попытка загрузки с OPUS...")
    logger.warning("Прямая загрузка с OPUS требует ручного скачивания файлов.")
    logger.info("Пожалуйста, скачайте файлы вручную с https://opus.nlpl.eu/WikiMatrix/corpus/version/WikiMatrix")
    logger.info("Или используйте альтернативный датасет через --use_alternative")
    
    raise Exception("Автоматическая загрузка с OPUS не реализована. Используйте альтернативный источник.")


def download_alternative_dataset(output_dir: str, max_samples: int = None, logger=None):
    """
    Загружает альтернативный датасет для тестирования.
    Использует OPUS-MT датасет или другие доступные источники.
    """
    try:
        from datasets import load_dataset
        
        logger.info("Загрузка альтернативного датасета (OPUS)...")
        
        # Используем OPUS dataset как альтернативу
        # Это параллельный корпус из OPUS
        try:
            # Пробуем загрузить OPUS dataset
            dataset = load_dataset("opus_books", "en-ru", split="train")
            logger.info("Используется OPUS Books dataset")
        except:
            # Если не получилось, пробуем другие варианты
            try:
                dataset = load_dataset("wmt19", "ru-en", split="train")
                logger.info("Используется WMT19 dataset (нужно будет перевернуть пары)")
                # Переворачиваем пары для ru-en -> en-ru
                dataset = dataset.map(lambda x: {
                    "translation": {
                        "en": x["translation"]["ru"],
                        "ru": x["translation"]["en"]
                    }
                })
            except:
                raise Exception("Не удалось загрузить альтернативные датасеты")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Загружено {len(dataset)} примеров")
        return dataset
    
    except ImportError:
        raise Exception("Библиотека 'datasets' не установлена")
    except Exception as e:
        raise Exception(f"Ошибка при загрузке альтернативного датасета: {e}")


def save_dataset(dataset, output_file: str, logger=None):
    """Сохраняет датасет в JSONL формат."""
    logger.info(f"Сохранение в {output_file}...")
    saved_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Сохранение"):
            try:
                # Обрабатываем разные форматы датасетов
                if "translation" in example:
                    source = example["translation"].get("en", "")
                    target = example["translation"].get("ru", "")
                elif "en" in example and "ru" in example:
                    source = example["en"]
                    target = example["ru"]
                else:
                    logger.warning(f"Неизвестный формат примера: {list(example.keys())}")
                    continue
                
                if not source or not target:
                    continue
                
                json.dump({
                    "source": source,
                    "target": target
                }, f, ensure_ascii=False)
                f.write('\n')
                saved_count += 1
            except Exception as e:
                logger.warning(f"Ошибка при сохранении примера: {e}")
                continue
    
    logger.info(f"Всего сохранено примеров: {saved_count}")
    return saved_count


def download_wikimatrix(
    output_dir: str = "data/raw", 
    max_samples: int = None, 
    logger=None,
    use_alternative: bool = False
):
    """
    Загружает датасет WikiMatrix EN-RU.
    
    Args:
        output_dir: Директория для сохранения данных
        max_samples: Максимальное количество примеров (None = все)
        logger: Логгер (опционально)
        use_alternative: Использовать альтернативный датасет если WikiMatrix недоступен
    """
    if logger is None:
        logger = setup_logger("download_data")
    
    logger.info("Загрузка датасета для машинного перевода EN-RU...")
    
    # Создаём директорию если её нет
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "wikimatrix_en_ru.jsonl")
    
    dataset = None
    
    # Пробуем разные способы загрузки
    if not use_alternative:
        try:
            dataset = download_from_huggingface(output_dir, max_samples, logger)
        except Exception as e:
            logger.warning(f"Не удалось загрузить через HuggingFace: {e}")
            if use_alternative:
                raise
            else:
                logger.info("Пробуем альтернативный источник...")
                use_alternative = True
    
    if use_alternative or dataset is None:
        try:
            dataset = download_alternative_dataset(output_dir, max_samples, logger)
        except Exception as e:
            logger.error(f"Не удалось загрузить альтернативный датасет: {e}")
            logger.error("\n" + "="*60)
            logger.error("РЕКОМЕНДАЦИИ:")
            logger.error("1. Установите зависимости: pip install datasets")
            logger.error("2. Проверьте интернет-соединение")
            logger.error("3. Скачайте датасет вручную:")
            logger.error("   - Список всех датасетов: https://opus.nlpl.eu/results/en&ru/corpus-result-table")
            logger.error("   - WikiMatrix: https://opus.nlpl.eu/WikiMatrix/corpus/version/WikiMatrix")
            logger.error("4. Или используйте --use_alternative для альтернативного датасета")
            logger.error("="*60)
            raise
    
    # Сохраняем датасет
    saved_count = save_dataset(dataset, output_file, logger)
    
    logger.info(f"Данные сохранены в {output_file}")
    logger.info(f"Всего сохранено примеров: {saved_count}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Загрузка датасета WikiMatrix EN-RU")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Директория для сохранения данных"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Максимальное количество примеров (для тестирования)"
    )
    parser.add_argument(
        "--use_alternative",
        action="store_true",
        help="Использовать альтернативный датасет (OPUS Books/WMT) если WikiMatrix недоступен"
    )
    
    args = parser.parse_args()
    
    logger = setup_logger("download_data")
    
    try:
        download_wikimatrix(
            args.output_dir, 
            args.max_samples, 
            logger,
            use_alternative=args.use_alternative
        )
        logger.info("Загрузка завершена успешно")
    except KeyboardInterrupt:
        logger.warning("Загрузка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

