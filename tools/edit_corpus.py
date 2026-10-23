import pandas as pd
import os
import sys
import shutil
from datetime import datetime

# Настройки путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.csv')

def create_backup(prefix="manual_backup"):
    """Создает уникальный бэкап с датой и временем."""
    if not os.path.exists(CORPUS_PATH):
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_filename = f"corpus_{prefix}_{timestamp}.csv"
    backup_path = os.path.join(DATA_DIR, backup_filename)
    shutil.copyfile(CORPUS_PATH, backup_path)
    print(f"[БЭКАП] Создана резервная копия: {backup_filename}")

def load_data():
    if not os.path.exists(CORPUS_PATH):
        print(f"[ОШИБКА] Файл не найден: {CORPUS_PATH}")
        sys.exit(1)
    # Сразу делаем бэкап перед загрузкой (на случай если что-то пойдет не так)
    create_backup("before_edit")
    return pd.read_csv(CORPUS_PATH, sep=';', encoding='utf-8-sig')

def save_data(df, original_df):
    """Сохраняет данные, предварительно делая уникальный бэкап."""
    if df.equals(original_df):
        return
    create_backup("after_edit")
    df.to_csv(CORPUS_PATH, sep=';', index=False, encoding='utf-8-sig')
    print("[УСПЕХ] Изменения сохранены в corpus.csv")

def parse_id_input(user_input):
    """Парсит строку вида '5', '5,10', '5-10', '5-10,15,20-22' в список ID."""
    ids = set()
    parts = user_input.replace(" ", "").split(",")
    
    for part in parts:
        if not part:
            continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end:
                    start, end = end, start
                ids.update(range(start, end + 1))
            except ValueError:
                print(f"[!] Неверный диапазон: {part}")
        else:
            try:
                ids.add(int(part))
            except ValueError:
                print(f"[!] Неверный ID: {part}")
                
    return sorted(list(ids))

def main():
    df = load_data()
    original_df = df.copy()
    
    print("="*60)
    print("ПАКЕТНЫЙ РЕДАКТОР КОРПУСА (v2.0 - Safe Edition)")
    print("Введите ID: '5', '5,10', '5-10', '5-10,15,20-25'")
    print("Введите 'q' в любой момент, чтобы сохранить изменения и выйти.")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nВведите ID (или 'q' для выхода): ").strip()
            
            if user_input.lower() == 'q':
                break
                
            target_ids = parse_id_input(user_input)
            
            if not target_ids:
                print("[!] Не удалось распознать ID.")
                continue
                
            # Проверяем, какие ID реально есть в базе
            existing_ids = [i for i in target_ids if i in df['id'].values]
            missing_ids = [i for i in target_ids if i not in df['id'].values]
            
            if missing_ids:
                print(f"[!] В базе не найдены ID: {missing_ids}")
                
            if not existing_ids:
                print("[!] Ни одного из указанных ID нет в базе.")
                continue
                
            # Показываем выбранные строки
            print("\n" + "-"*50)
            print(f"Выбрано строк: {len(existing_ids)}")
            
            display_ids = existing_ids if len(existing_ids) <= 3 else existing_ids[:3] + ['...'] + [existing_ids[-1]]
            
            for msg_id in display_ids:
                if msg_id == '...':
                    print("...")
                    continue
                row = df[df['id'] == msg_id].iloc[0]
                # Обрезаем текст для вывода, чтобы не ломать консоль
                short_text = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
                print(f"ID {row['id']}: [{row['intent']} / {row['urgency']}] {short_text}")
            print("-" * 50)
            
            # Меню действий
            print("Что делаем с выбранными строками?")
            print("1. Изменить текст (заменит текст на введенный)")
            print("2. Изменить категорию (category)")
            print("3. Изменить интент (intent)")
            print("4. Изменить urgency")
            print("5. УДАЛИТЬ выбранные сообщения (с пересчетом ID)")
            print("Enter. Оставить как есть")
            
            action = input("Выбор (1-5): ").strip()
            
            if action == '1':
                new_text = input("Новый текст (применится ко всем выбранным): ").strip()
                if new_text:
                    df.loc[df['id'].isin(existing_ids), 'text'] = new_text
                    print(f"[ОК] Текст обновлен у {len(existing_ids)} строк.")
                    
            elif action == '2':
                new_cat = input("Новая категория (SOS/Информационное/Запрос/Тест): ").strip()
                if new_cat:
                    df.loc[df['id'].isin(existing_ids), 'category'] = new_cat
                    print(f"[ОК] Категория обновлена у {len(existing_ids)} строк.")
                    
            elif action == '3':
                new_intent = input("Новый интент: ").strip()
                if new_intent:
                    df.loc[df['id'].isin(existing_ids), 'intent'] = new_intent
                    print(f"[ОК] Интент обновлен у {len(existing_ids)} строк.")
                    
            elif action == '4':
                new_urg = input("Новый urgency (1-4): ").strip()
                if new_urg in ['1', '2', '3', '4']:
                    df.loc[df['id'].isin(existing_ids), 'urgency'] = int(new_urg)
                    print(f"[ОК] Urgency обновлен у {len(existing_ids)} строк.")
                else:
                    print("[!] Ошибка: urgency должно быть числом от 1 до 4.")
                    
            elif action == '5':
                confirm = input(f"Вы уверены, что хотите УДАЛИТЬ {len(existing_ids)} сообщений? (y/n): ").strip().lower()
                if confirm == 'y':
                    df = df[~df['id'].isin(existing_ids)].copy()
                    # Пересчитываем ID для всех оставшихся строк
                    df['id'] = range(1, len(df) + 1)
                    print(f"[УДАЛЕНО] Удалено {len(existing_ids)} строк. ID пересчитаны.")
                else:
                    print("[ОК] Удаление отменено.")
            else:
                print("[ОК] Изменений нет.")
                
        except KeyboardInterrupt:
            print("\nПрервано. Выход.")
            break
        except Exception as e:
            print(f"[ОШИБКА] {e}")
            
    save_data(df, original_df)
    print("\nДо свидания! Не забудьте проверить папку data на наличие бэкапов.")

if __name__ == "__main__":
    main()