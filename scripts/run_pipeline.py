"""
Главный скрипт для запуска всего пайплайна:
1. Загрузка данных
2. Предобработка
3. Обучение
4. Оценка
"""

import os
import sys
import argparse
import subprocess

def run_command(cmd, description):
    """Запускает команду и выводит результат."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Команда: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ОШИБКА: Команда завершилась с кодом {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Запуск полного пайплайна обучения модели"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Пропустить загрузку данных"
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Пропустить предобработку"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Пропустить обучение"
    )
    parser.add_argument(
        "--skip_evaluate",
        action="store_true",
        help="Пропустить оценку"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Максимальное количество примеров для тестирования"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-ru",
        help="Название pretrained модели"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Размер батча"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Количество эпох"
    )
    parser.add_argument(
        "--use_alternative",
        action="store_true",
        help="Использовать альтернативный датасет если WikiMatrix недоступен"
    )
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(base_dir, "scripts")
    
    # 1. Загрузка данных
    if not args.skip_download:
        cmd = [
            sys.executable,
            os.path.join(scripts_dir, "download_data.py")
        ]
        if args.max_samples:
            cmd.extend(["--max_samples", str(args.max_samples)])
        if args.use_alternative:
            cmd.append("--use_alternative")
        
        if not run_command(cmd, "ШАГ 1: Загрузка данных"):
            return
    
    # 2. Предобработка
    if not args.skip_preprocess:
        cmd = [
            sys.executable,
            os.path.join(scripts_dir, "preprocess.py"),
            "--input_file", "data/raw/wikimatrix_en_ru.jsonl",
            "--output_file", "data/processed/preprocessed.jsonl",
            "--split"
        ]
        
        if not run_command(cmd, "ШАГ 2: Предобработка данных"):
            return
    
    # 3. Обучение
    if not args.skip_train:
        cmd = [
            sys.executable,
            os.path.join(scripts_dir, "train.py"),
            "--train_file", "data/processed/train.jsonl",
            "--val_file", "data/processed/val.jsonl",
            "--model_name", args.model_name,
            "--batch_size", str(args.batch_size),
            "--num_epochs", str(args.num_epochs)
        ]
        
        if not run_command(cmd, "ШАГ 3: Обучение модели"):
            return
    
    # 4. Оценка
    if not args.skip_evaluate:
        # Находим последнюю модель
        checkpoint_dir = os.path.join(base_dir, "models", "checkpoint")
        final_model_dir = os.path.join(checkpoint_dir, "final_model")
        
        if not os.path.exists(final_model_dir):
            print(f"Модель не найдена в {final_model_dir}")
            print("Попробуйте указать путь к модели вручную")
            return
        
        cmd = [
            sys.executable,
            os.path.join(scripts_dir, "evaluate.py"),
            "--model_dir", final_model_dir,
            "--test_file", "data/processed/test.jsonl"
        ]
        
        if not run_command(cmd, "ШАГ 4: Оценка модели"):
            return
    
    print("\n" + "="*60)
    print("ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО!")
    print("="*60)


if __name__ == "__main__":
    main()

