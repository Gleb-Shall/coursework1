# Руководство по получению математических данных

## Проблема

Текущий датасет (WikiMatrix/OPUS Books) содержит в основном художественную литературу, а не математические тексты. Математических текстов очень мало (~1-4%).

## Решения

### 1. Использовать улучшенную фильтрацию (рекомендуется для начала)

Запустите скрипт для поиска математических текстов в существующем датасете:

```bash
# С более строгой фильтрацией
python3 scripts/find_math_texts.py \
    --input_file data/raw/wikimatrix_en_ru.jsonl \
    --output_file data/raw/math_filtered.jsonl \
    --min_score 0.5
```

Затем используйте отфильтрованные данные:

```bash
python3 scripts/preprocess.py \
    --input_file data/raw/math_filtered.jsonl \
    --output_file data/processed/math_preprocessed.jsonl \
    --no_filter_math  # Фильтрация уже выполнена
```

### 2. Использовать специализированные математические датасеты

#### Вариант A: OPUS Math (если доступен)

Попробуйте найти математические корпуса на https://opus.nlpl.eu/

#### Вариант B: Создать свой датасет из Wikipedia

1. Найдите математические статьи на Wikipedia:
   - https://en.wikipedia.org/wiki/Category:Mathematics
   - https://ru.wikipedia.org/wiki/Категория:Математика

2. Используйте инструменты для извлечения и выравнивания:
   - https://github.com/facebookresearch/LASER (для выравнивания)
   - https://github.com/Helsinki-NLP/OPUS-MT-train (для создания параллельного корпуса)

#### Вариант C: Использовать arXiv статей

1. Скачайте статьи с arXiv.org (раздел Mathematics)
2. Извлеките тексты и формулы
3. Используйте Google Translate API или другие сервисы для перевода (осторожно с качеством)

### 3. Смешанный подход

1. Используйте небольшое количество отфильтрованных математических текстов
2. Дополните общим датасетом (WikiMatrix) для базового перевода
3. Дообучите модель на математических примерах

```bash
# Шаг 1: Найти математические тексты
python3 scripts/find_math_texts.py \
    --input_file data/raw/wikimatrix_en_ru.jsonl \
    --output_file data/raw/math_texts.jsonl \
    --min_score 0.5

# Шаг 2: Предобработать математические тексты
python3 scripts/preprocess.py \
    --input_file data/raw/math_texts.jsonl \
    --output_file data/processed/math_train.jsonl \
    --no_filter_math

# Шаг 3: Использовать для обучения
python3 scripts/train.py \
    --train_file data/processed/math_train.jsonl \
    --val_file data/processed/val.jsonl
```

### 4. Использовать синтетические данные

Создайте простые математические примеры вручную:

```python
examples = [
    {
        "source": "Let $f(x) = x^2$ be a function. Then $f'(x) = 2x$.",
        "target": "Пусть $f(x) = x^2$ — функция. Тогда $f'(x) = 2x$."
    },
    {
        "source": "The integral of $x^2$ is $\\frac{x^3}{3} + C$.",
        "target": "Интеграл от $x^2$ равен $\\frac{x^3}{3} + C$."
    },
    # ... больше примеров
]
```

### 5. Использовать готовые математические корпуса

Поищите в научных публикациях:
- Papers with Code: https://paperswithcode.com/
- ACL Anthology: https://aclanthology.org/
- arXiv: https://arxiv.org/list/math/recent

## Рекомендации для курсовой работы

Для курсовой работы (2 курс бакалавриата) достаточно:

1. **Использовать улучшенную фильтрацию** из существующего датасета
2. **Создать 50-100 примеров вручную** с математическими формулами
3. **Объединить** отфильтрованные данные + ручные примеры
4. **Дообучить модель** на этом небольшом, но качественном датасете

Это покажет понимание задачи и умение работать с данными, что достаточно для курсовой.

## Быстрый старт

```bash
# 1. Найти математические тексты
python3 scripts/find_math_texts.py \
    --input_file data/raw/wikimatrix_en_ru.jsonl \
    --output_file data/raw/math_filtered.jsonl \
    --min_score 0.5

# 2. Проверить результат
head -20 data/raw/math_filtered.jsonl

# 3. Если результат хороший, использовать для обучения
python3 scripts/preprocess.py \
    --input_file data/raw/math_filtered.jsonl \
    --output_file data/processed/math_preprocessed.jsonl \
    --no_filter_math \
    --split
```

## Примечания

- Математических параллельных корпусов EN-RU очень мало
- Большинство математических текстов в интернете на английском
- Для качественного перевода математики может потребоваться ручная работа или специализированные инструменты

