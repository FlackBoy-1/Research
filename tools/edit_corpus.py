import pandas as pd
import os
import sys
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.csv')
VIEW_PATH = os.path.join(DATA_DIR, 'filtered_view.txt')

def create_backup(prefix="manual_backup"):
    if not os.path.exists(CORPUS_PATH):
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_filename = f"corpus_{prefix}_{timestamp}.csv"
    backup_path = os.path.join(DATA_DIR, backup_filename)
    shutil.copyfile(CORPUS_PATH, backup_path)
    print(f"[БЭКАП] Создана копия: {backup_filename}")

def load_data():
    if not os.path.exists(CORPUS_PATH):
        print(f"[ОШИБКА] Файл не найден: {CORPUS_PATH}")
        sys.exit(1)
    create_backup("before_edit")
    return pd.read_csv(CORPUS_PATH, sep=';', encoding='utf-8-sig')

def save_data(df, original_df):
    if df.equals(original_df):
        return
    create_backup("after_edit")
    # Пересчитываем ID ТОЛЬКО перед сохранением
    df['id'] = range(1, len(df) + 1)
    df.to_csv(CORPUS_PATH, sep=';', index=False, encoding='utf-8-sig')
    print("[УСПЕХ] Изменения сохранены в corpus.csv (ID пересчитаны от 1 до N)")

def parse_id_input(user_input):
    ids = set()
    parts = user_input.replace(" ", "").split(",")
    for part in parts:
        if not part: continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end: start, end = end, start
                ids.update(range(start, end + 1))
            except ValueError:
                print(f"[!] Неверный диапазон: {part}")
        else:
            try:
                ids.add(int(part))
            except ValueError:
                print(f"[!] Неверный ID: {part}")
    return sorted(list(ids))

def filter_data(df):
    filtered_df = df.copy()
    
    print("\n--- НАСТРОЙКА ФИЛЬТРА (Enter, чтобы пропустить) ---")
    
    source = input("Источник данных (1 - только реальные, 2 - только синтетика, Enter - все): ").strip()
    if source == '1' and 'is_synthetic' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_synthetic'] == 0]
        print("  -> Фильтр: только реальные сообщения")
    elif source == '2' and 'is_synthetic' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_synthetic'] == 1]
        print("  -> Фильтр: только синтетические сообщения")
    else:
        print("  -> Фильтр: все сообщения")

    cat = input("Категория (SOS/Информационное/Запрос/Тест): ").strip()
    if cat:
        filtered_df = filtered_df[filtered_df['category'] == cat]
        
    intent = input("Интент (например, позиция): ").strip()
    if intent:
        filtered_df = filtered_df[filtered_df['intent'] == intent]
        
    urg = input("Срочность (1-4): ").strip()
    if urg:
        try:
            filtered_df = filtered_df[filtered_df['urgency'] == int(urg)]
        except ValueError:
            print("[!] Urgency должно быть числом. Пропускаем фильтр.")
            
    return filtered_df

def export_view(filtered_df):
    if filtered_df.empty:
        return
    with open(VIEW_PATH, 'w', encoding='utf-8') as f:
        f.write(f"ВЫГРУЗКА КОРПУСА (Найдено строк: {len(filtered_df)})\n")
        f.write("=" * 80 + "\n\n")
        for _, row in filtered_df.iterrows():
            synth_mark = "[СИНТ]" if row.get('is_synthetic') == 1 else "[РЕАЛ]"
            f.write(f"ID: {row['id']} | {synth_mark} | Интент: {row['intent']} | Срочность: {row['urgency']} | Категория: {row['category']}\n")
            f.write(f"ТЕКСТ: {row['text']}\n")
            f.write("-" * 80 + "\n")

def main():
    df = load_data()
    original_df = df.copy()
    
    print("="*60)
    print("РЕДАКТОР КОРПУСА (v3.4 - Full Source Control + Synthetic Toggle)")
    print("ID не меняются во время сессии. Пересчет только при выходе.")
    print("="*60)
    
    current_filtered = pd.DataFrame()
    
    while True:
        try:
            if current_filtered.empty:
                print("\n--- ГЛАВНОЕ МЕНЮ ---")
                print("1. Настроить фильтр и выгрузить в файл")
                print("2. Ввести ID вручную (без фильтра)")
                print("q. Сохранить и выйти")
                choice = input("Выбор: ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == '1':
                    current_filtered = filter_data(df)
                    if current_filtered.empty:
                        print("\n[!] По вашему запросу ничего не найдено.")
                        current_filtered = pd.DataFrame()
                        continue
                    
                    export_view(current_filtered)
                    print(f"\n[+] Найдено сообщений: {len(current_filtered)}")
                    print(f"[+] Полный текст выгружен в файл: data/filtered_view.txt")
                    
                    print("\n--- Превью (первые 3) ---")
                    for _, row in current_filtered.head(3).iterrows():
                        synth_mark = "[СИНТ]" if row.get('is_synthetic') == 1 else "[РЕАЛ]"
                        short_text = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
                        print(f"ID {row['id']} {synth_mark}: {short_text}")

                elif choice == '2':
                    current_filtered = df.copy() 
                else:
                    print("[!] Неверный ввод.")
                    continue
            else:
                print("\n" + "-"*50)
                print("Введите ID для изменения, 'c' чтобы сбросить фильтр, или 'q' для выхода.")
                user_input = input("ID (или 'c'/'q'): ").strip()
                
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 'c':
                    current_filtered = pd.DataFrame()
                    continue
                    
                target_ids = parse_id_input(user_input)
                existing_ids = [i for i in target_ids if i in current_filtered['id'].values]
                
                if not existing_ids:
                    print("[!] В текущем фильтре нет таких ID.")
                    continue
                    
                print(f"\nВыбрано строк: {len(existing_ids)}")
                for msg_id in existing_ids[:2]:
                    row = df[df['id'] == msg_id].iloc[0]
                    synth_mark = "[СИНТ]" if row.get('is_synthetic') == 1 else "[РЕАЛ]"
                    print(f"ID {row['id']} {synth_mark}: [{row['intent']} / {row['urgency']}] {row['text'][:60]}...")
                    
                print("1. Изменить ТЕКСТ (только для 1 ID!)")
                print("2. Изменить КАТЕГОРИЮ")
                print("3. Изменить ИНТЕНТ")
                print("4. Изменить URGENCY")
                print("5. УДАЛИТЬ выбранные")
                print("6. Переключить флаг СИНТЕТИКИ (0/1)")
                print("Enter. Отмена")
                
                action = input("Действие: ").strip()
                
                if action == '1':
                    if len(existing_ids) > 1:
                        print("[!] ОШИБКА: Нельзя менять текст у группы.")
                    else:
                        new_text = input("Новый текст: ").strip()
                        if new_text:
                            df.loc[df['id'] == existing_ids[0], 'text'] = new_text
                            print("[ОК] Текст обновлен.")
                            
                elif action == '2':
                    new_cat = input("Новая категория: ").strip()
                    if new_cat:
                        df.loc[df['id'].isin(existing_ids), 'category'] = new_cat
                        print(f"[ОК] Категория обновлена.")
                        
                elif action == '3':
                    new_intent = input("Новый интент: ").strip()
                    if new_intent:
                        df.loc[df['id'].isin(existing_ids), 'intent'] = new_intent
                        print(f"[ОК] Интент обновлен.")
                        
                elif action == '4':
                    new_urg = input("Новый urgency (1-4): ").strip()
                    if new_urg in ['1', '2', '3', '4']:
                        df.loc[df['id'].isin(existing_ids), 'urgency'] = int(new_urg)
                        print(f"[ОК] Urgency обновлен.")

                elif action == '6':
                    new_synth = input("Новое значение is_synthetic (0 - реал, 1 - синт): ").strip()
                    if new_synth in ['0', '1']:
                        df.loc[df['id'].isin(existing_ids), 'is_synthetic'] = int(new_synth)
                        current_filtered.loc[current_filtered['id'].isin(existing_ids), 'is_synthetic'] = int(new_synth)
                        print(f"[ОК] Флаг is_synthetic обновлен на {new_synth} для {len(existing_ids)} строк.")
                    else:
                        print("[!] Неверный ввод. Введите 0 или 1.")
                        
                elif action == '5':
                    confirm = input(f"УДАЛИТЬ {len(existing_ids)} сообщений? (y/n): ").strip().lower()
                    if confirm == 'y':
                        df = df[~df['id'].isin(existing_ids)].copy()
                        current_filtered = current_filtered[~current_filtered['id'].isin(existing_ids)].copy()
                        print(f"[УДАЛЕНО] Удалено {len(existing_ids)} строк. ID остальных не изменены.")
                else:
                    print("[ОК] Отмена.")
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ОШИБКА] {e}")
            
    save_data(df, original_df)

if __name__ == "__main__":
    main()