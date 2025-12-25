# Дообучение трансформерной модели для машинного перевода математических текстов (EN ↔ RU)

Курсовая работа по машинному обучению и NLP.

## Описание

Проект направлен на дообучение трансформерной модели для качественного перевода математических текстов между английским и русским языками с сохранением:
- Математического смысла
- Корректности формул (LaTeX)
- Структуры доказательств и определений
- Терминологической точности

## Структура проекта

```
coursework1/
├── data/                    # Данные (датасеты, предобработанные файлы)
│   ├── raw/                # Исходные данные
│   └── processed/          # Предобработанные данные
├── models/                  # Сохранённые модели
├── scripts/                 # Основные скрипты
│   ├── download_data.py    # Загрузка датасета
│   ├── preprocess.py       # Предобработка данных
│   ├── train.py            # Обучение модели
│   ├── evaluate.py         # Оценка качества
│   ├── translate.py        # Быстрый перевод
│   ├── analyze_data.py     # Анализ датасета
│   ├── check_setup.py      # Проверка окружения
│   ├── compare_models.py   # Сравнение моделей
│   └── run_pipeline.py     # Запуск всего пайплайна
├── utils/                   # Вспомогательные модули
│   ├── formula_handler.py  # Работа с формулами
│   ├── metrics.py          # Метрики оценки
│   ├── logger.py           # Логирование
│   ├── validation.py       # Валидация данных
│   └── visualization.py    # Визуализация результатов
├── results/                 # Результаты экспериментов
└── logs/                    # Логи выполнения
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Быстрый старт (весь пайплайн)

```bash
# Запуск всего пайплайна одной командой
python scripts/run_pipeline.py

# Для тестирования с меньшим количеством данных
python scripts/run_pipeline.py --max_samples 10000
```

### Пошаговое выполнение

1. **Загрузка данных:**
```bash
# Стандартная загрузка
python scripts/download_data.py

# С ограничением количества примеров (для тестирования)
python scripts/download_data.py --max_samples 50000

# С альтернативным источником (если WikiMatrix недоступен)
python scripts/download_data.py --use_alternative --max_samples 50000
```

2. **Предобработка:**
```bash
python scripts/preprocess.py

# С разделением на train/val/test
python scripts/preprocess.py --split

# Без фильтрации математических текстов
python scripts/preprocess.py --no_filter_math
```

3. **Обучение:**
```bash
python scripts/train.py \
    --train_file data/processed/train.jsonl \
    --val_file data/processed/val.jsonl \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 2e-5
```

4. **Оценка:**
```bash
python scripts/evaluate.py \
    --model_dir models/checkpoint/final_model \
    --test_file data/processed/test.jsonl
```

5. **Перевод текста:**
```bash
python scripts/translate.py \
    --model_dir models/checkpoint/final_model \
    --text "Let $f(x)$ be a continuous function."
```

6. **Анализ данных:**
```bash
python scripts/analyze_data.py \
    --data_file data/processed/preprocessed.jsonl
```

7. **Проверка окружения:**
```bash
python scripts/check_setup.py
```

8. **Сравнение моделей:**
```bash
python scripts/compare_models.py \
    --model_dirs models/checkpoint/final_model models/baseline_model
```

## Модель

Базовая модель: **MarianMT** (Helsinki-NLP/opus-mt-en-ru)

### Альтернативные модели

Можно использовать другие модели, изменив параметр `--model_name`:
- `Helsinki-NLP/opus-mt-en-ru` (по умолчанию, рекомендуется)
- `facebook/mbart-large-50-many-to-many-mmt` (требует больше памяти)
- `t5-base` (требует адаптации для перевода)

## Датасет

**WikiMatrix EN-RU** с фильтрацией математических текстов.

### Загрузка данных

Датасет можно загрузить несколькими способами:

1. **Автоматическая загрузка** (через скрипт):
```bash
python scripts/download_data.py
```

2. **С альтернативным источником** (если WikiMatrix недоступен):
```bash
python scripts/download_data.py --use_alternative
```

3. **Ручная загрузка с OPUS**:
   - Список всех датасетов: https://opus.nlpl.eu/results/en&ru/corpus-result-table
   - WikiMatrix: https://opus.nlpl.eu/WikiMatrix/corpus/version/WikiMatrix
   - Выберите языковую пару English-Russian
   - Скачайте файлы в формате TMX или Moses
   - Конвертируйте в JSONL формат (см. DATA_SOURCES.md)

### Фильтрация данных

Фильтрация математических текстов происходит по:
- Наличию LaTeX формул (`$...$`, `$$...$$`)
- Математическим ключевым словам (theorem, proof, function, интеграл, производная и т.д.)

## Требования

- Python 3.8+
- PyTorch 2.0+
- GPU с 8-16 GB VRAM (рекомендуется)
- CUDA (для GPU ускорения)

## Конфигурация

Основные параметры можно изменить в файле `config.py`:
- Размер батча
- Learning rate
- Количество эпох
- Параметры предобработки

## Результаты

Результаты оценки сохраняются в:
- `results/evaluation_results.json` - метрики (BLEU, chrF, Formula Preservation)
- `results/evaluation_results_examples.json` - примеры переводов

## Особенности реализации

1. **Обработка формул**: Формулы сохраняются в тексте, модель учится переводить контекст вокруг них
2. **Фильтрация данных**: Автоматический отбор математических текстов
3. **Специализированные метрики**: Formula Preservation для оценки сохранения формул
4. **Воспроизводимость**: Фиксированный seed для разделения данных

## Структура данных

Формат входных данных (JSONL):
```json
{"source": "Let $f(x)$ be a continuous function.", "target": "Пусть $f(x)$ — непрерывная функция."}
```

## Лицензия

Проект создан в образовательных целях для курсовой работы.

