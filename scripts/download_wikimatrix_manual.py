"""
Инструкции и скрипт для ручной загрузки WikiMatrix с OPUS.
WikiMatrix содержит реальные статьи из Wikipedia, включая математические.
"""

import os
import json
import argparse
from xml.etree import ElementTree as ET


def convert_tmx_to_jsonl(tmx_file: str, output_file: str):
    """Конвертирует TMX файл в JSONL формат."""
    print(f"Конвертация {tmx_file} в {output_file}...")
    
    tree = ET.parse(tmx_file)
    root = tree.getroot()
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for tu in root.findall('.//tu'):
            try:
                # Ищем тексты на английском и русском
                en_seg = None
                ru_seg = None
                
                for tuv in tu.findall('.//tuv'):
                    lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang', '').lower()
                    seg = tuv.find('.//seg')
                    if seg is not None and seg.text:
                        if lang == 'en':
                            en_seg = seg.text.strip()
                        elif lang == 'ru':
                            ru_seg = seg.text.strip()
                
                if en_seg and ru_seg:
                    json.dump({
                        "source": en_seg,
                        "target": ru_seg
                    }, f, ensure_ascii=False)
                    f.write('\n')
                    count += 1
            except Exception as e:
                continue
    
    print(f"Конвертировано {count} примеров")
    return count


def convert_moses_to_jsonl(moses_file_en: str, moses_file_ru: str, output_file: str):
    """Конвертирует два Moses файла (EN и RU) в JSONL."""
    print(f"Конвертация Moses файлов в {output_file}...")
    
    count = 0
    with open(moses_file_en, 'r', encoding='utf-8') as f_en, \
         open(moses_file_ru, 'r', encoding='utf-8') as f_ru, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for en_line, ru_line in zip(f_en, f_ru):
            en_text = en_line.strip()
            ru_text = ru_line.strip()
            
            if en_text and ru_text:
                json.dump({
                    "source": en_text,
                    "target": ru_text
                }, f_out, ensure_ascii=False)
                f_out.write('\n')
                count += 1
    
    print(f"Конвертировано {count} примеров")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация WikiMatrix из OPUS в JSONL"
    )
    parser.add_argument(
        "--tmx_file",
        type=str,
        help="TMX файл для конвертации"
    )
    parser.add_argument(
        "--moses_en",
        type=str,
        help="Moses файл с английскими текстами"
    )
    parser.add_argument(
        "--moses_ru",
        type=str,
        help="Moses файл с русскими текстами"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/raw/wikimatrix_from_opus.jsonl",
        help="Выходной JSONL файл"
    )
    
    args = parser.parse_args()
    
    if args.tmx_file:
        convert_tmx_to_jsonl(args.tmx_file, args.output_file)
    elif args.moses_en and args.moses_ru:
        convert_moses_to_jsonl(args.moses_en, args.moses_ru, args.output_file)
    else:
        print("="*60)
        print("ИНСТРУКЦИЯ ПО ЗАГРУЗКЕ WIKIMATRIX С OPUS")
        print("="*60)
        print("\n1. Перейдите на:")
        print("   https://opus.nlpl.eu/WikiMatrix/corpus/version/WikiMatrix")
        print("\n2. Выберите языковую пару: English → Russian")
        print("\n3. Скачайте файлы в формате:")
        print("   - TMX (рекомендуется) ИЛИ")
        print("   - Moses (два файла: .en и .ru)")
        print("\n4. Поместите файлы в data/raw/")
        print("\n5. Запустите конвертацию:")
        print("   Для TMX:")
        print("   python3 scripts/download_wikimatrix_manual.py \\")
        print("       --tmx_file data/raw/wikimatrix.en-ru.tmx \\")
        print("       --output_file data/raw/wikimatrix_real.jsonl")
        print("\n   Для Moses:")
        print("   python3 scripts/download_wikimatrix_manual.py \\")
        print("       --moses_en data/raw/wikimatrix.en-ru.en \\")
        print("       --moses_ru data/raw/wikimatrix.en-ru.ru \\")
        print("       --output_file data/raw/wikimatrix_real.jsonl")
        print("\n6. После конвертации используйте:")
        print("   python3 scripts/preprocess.py \\")
        print("       --input_file data/raw/wikimatrix_real.jsonl \\")
        print("       --output_file data/processed/wikimatrix_preprocessed.jsonl \\")
        print("       --split")
        print("="*60)


if __name__ == "__main__":
    main()

