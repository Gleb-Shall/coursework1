# Источники данных

## WikiMatrix

**Основной источник**: [OPUS WikiMatrix](https://opus.nlpl.eu/WikiMatrix/corpus/version/WikiMatrix)

WikiMatrix - это параллельный корпус из Wikipedia, содержащий 135M параллельных предложений в 1620 языковых парах.

**Список всех доступных датасетов EN-RU**: [OPUS EN-RU Corpora](https://opus.nlpl.eu/results/en&ru/corpus-result-table)

На этой странице можно найти все доступные параллельные корпуса для пары английский-русский, включая:
- WikiMatrix
- OPUS Books
- GNOME
- Ubuntu
- KDE
- TED Talks
- И многие другие

### Автоматическая загрузка

Скрипт `download_data.py` пытается загрузить данные автоматически через HuggingFace datasets. Если это не работает, используйте флаг `--use_alternative` для загрузки альтернативного датасета.

### Ручная загрузка

**Вариант 1: Через список всех датасетов**
1. Перейдите на https://opus.nlpl.eu/results/en&ru/corpus-result-table
2. Выберите нужный датасет (например, WikiMatrix)
3. Выберите языковую пару: **English (source)** → **Russian (target)**
4. Скачайте файлы в формате TMX или Moses

**Вариант 2: Прямая ссылка на WikiMatrix**
1. Перейдите на https://opus.nlpl.eu/WikiMatrix/corpus/version/WikiMatrix
2. Выберите языковую пару: **English (source)** → **Russian (target)**
3. Скачайте файлы в формате TMX или Moses

**Конвертация в JSONL:**
4. Конвертируйте скачанные файлы в JSONL формат:

```python
import json
from xml.etree import ElementTree as ET

# Для TMX формата
tree = ET.parse('downloaded_file.tmx')
root = tree.getroot()

with open('data/raw/wikimatrix_en_ru.jsonl', 'w', encoding='utf-8') as f:
    for tu in root.findall('.//tu'):
        en_text = tu.find('.//tuv[@xml:lang="en"]/seg').text
        ru_text = tu.find('.//tuv[@xml:lang="ru"]/seg').text
        json.dump({"source": en_text, "target": ru_text}, f, ensure_ascii=False)
        f.write('\n')
```

## Альтернативные источники

Если WikiMatrix недоступен, скрипт автоматически использует альтернативные датасеты:

1. **OPUS Books** - параллельный корпус книг
2. **WMT19** - датасет конференции WMT (нужно перевернуть пары)

### Использование альтернативных источников

```bash
python scripts/download_data.py --use_alternative
```

## Другие возможные источники

Для курсовой работы можно также использовать:

1. **OPUS GNOME/Ubuntu/KDE** - технические тексты
   - URL: https://opus.nlpl.eu/
   - Содержат формальную лексику, полезны для терминологии

2. **arXiv Math** (самостоятельно собранный)
   - Математические статьи на английском
   - Можно использовать как monolingual corpus

3. **TED Talks** - выступления TED
   - URL: https://opus.nlpl.eu/TED2013.php
   - Хорошее качество переводов

## Формат данных

Все данные должны быть в формате JSONL:

```json
{"source": "English text", "target": "Русский текст"}
{"source": "Another sentence.", "target": "Другое предложение."}
```

## Рекомендации

- Для тестирования используйте `--max_samples 10000`
- Для полного обучения рекомендуется минимум 50-100k пар предложений
- Идеально: 300k - 1M пар для качественного обучения

