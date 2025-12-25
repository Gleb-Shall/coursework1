# Быстрый старт

## 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 2. Проверка окружения

```bash
python scripts/check_setup.py
```

Эта команда проверит:
- Установлены ли все необходимые пакеты
- Доступен ли GPU
- Правильно ли организована структура проекта

## 3. Запуск полного пайплайна

### Вариант А: Полный запуск (для тестирования с малым количеством данных)

```bash
python scripts/run_pipeline.py --max_samples 10000
```

### Вариант Б: Пошаговое выполнение

#### Шаг 1: Загрузка данных
```bash
python scripts/download_data.py --max_samples 50000
```

#### Шаг 2: Анализ данных (опционально)
```bash
python scripts/analyze_data.py --data_file data/raw/wikimatrix_en_ru.jsonl
```

#### Шаг 3: Предобработка
```bash
python scripts/preprocess.py \
    --input_file data/raw/wikimatrix_en_ru.jsonl \
    --output_file data/processed/preprocessed.jsonl \
    --split
```

#### Шаг 4: Обучение
```bash
python scripts/train.py \
    --train_file data/processed/train.jsonl \
    --val_file data/processed/val.jsonl \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 2e-5
```

#### Шаг 5: Оценка
```bash
python scripts/evaluate.py \
    --model_dir models/checkpoint/final_model \
    --test_file data/processed/test.jsonl
```

## 4. Использование обученной модели

### Перевод одного текста
```bash
python scripts/translate.py \
    --model_dir models/checkpoint/final_model \
    --text "Let $f(x) = x^2$ be a function."
```

### Сравнение нескольких моделей
```bash
python scripts/compare_models.py \
    --model_dirs models/checkpoint/final_model models/baseline_model \
    --test_file data/processed/test.jsonl
```

## 5. Просмотр результатов

Результаты сохраняются в:
- `results/evaluation_results.json` - метрики
- `results/evaluation_results_examples.json` - примеры переводов
- `results/analysis/` - анализ данных
- `logs/` - логи выполнения

## Параметры обучения

Основные параметры можно изменить в `config.py` или через аргументы командной строки:

- `--batch_size`: Размер батча (по умолчанию: 8)
- `--learning_rate`: Learning rate (по умолчанию: 2e-5)
- `--num_epochs`: Количество эпох (по умолчанию: 3)
- `--max_length`: Максимальная длина последовательности (по умолчанию: 512)

## Требования к системе

- **Минимум**: CPU, 8GB RAM
- **Рекомендуется**: GPU с 8-16GB VRAM, CUDA
- Python 3.8+

## Решение проблем

### Ошибка "Out of memory"
- Уменьшите `--batch_size`
- Увеличьте `--gradient_accumulation_steps`
- Отключите `--no_fp16` (если был включен)

### Медленное обучение
- Убедитесь, что используется GPU
- Проверьте `python scripts/check_setup.py`
- Уменьшите `--max_length`

### Ошибки при загрузке данных
- Проверьте интернет-соединение
- Попробуйте уменьшить `--max_samples` для тестирования

## Дополнительная информация

Подробная документация в `README.md` и комментариях в коде.

