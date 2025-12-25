"""
Конфигурационный файл для проекта.
"""

# Модель
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
# Альтернативы:
# MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"  # mBART (требует больше памяти)
# MODEL_NAME = "t5-base"  # T5 (нужна адаптация для перевода)

# Пути к данным
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Параметры предобработки
MIN_SENTENCE_LENGTH = 5
MAX_SENTENCE_LENGTH = 512
FILTER_MATH_TEXTS = True

# Параметры обучения
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 1
FP16 = True  # Mixed precision (требует GPU с поддержкой)

# Параметры оценки
EVAL_BATCH_SIZE = 8
MAX_LENGTH = 512

# Математические термины для оценки (пример)
MATH_TERMS = {
    "theorem": ["теорема"],
    "proof": ["доказательство"],
    "lemma": ["лемма"],
    "function": ["функция"],
    "derivative": ["производная"],
    "integral": ["интеграл"],
    "limit": ["предел"],
    "matrix": ["матрица"],
    "vector": ["вектор"],
    "equation": ["уравнение"]
}

