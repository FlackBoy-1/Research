import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, 'data', 'corpus.csv')

def check_balance():
    if not os.path.exists(CORPUS_PATH):
        print(f"[ОШИБКА] Файл не найден: {CORPUS_PATH}")
        return

    df = pd.read_csv(CORPUS_PATH, sep=';', encoding='utf-8-sig')
    total = len(df)
    
    # --- СЧЕТЧИК СИНТЕТИКИ ---
    # Если колонка пустая или ее нет, считаем, что синтетики 0
    if 'is_synthetic' in df.columns:
        synthetic_count = df['is_synthetic'].sum()
    else:
        synthetic_count = 0
        
    real_count = total - synthetic_count
    syn_percent = (synthetic_count / total) * 100 if total > 0 else 0

    print("=" * 50)
    print(f"ВСЕГО СООБЩЕНИЙ В БАЗЕ: {total}")
    print(f"  Реальных:  {real_count} ({(real_count/total)*100:.1f}%)")
    print(f"  Синтетики: {synthetic_count} ({syn_percent:.1f}%)")
    print("=" * 50)

    # 1. Статистика по категориям
    print("\n--- СТАТИСТИКА ПО КАТЕГОРИЯМ (category) ---")
    cat_counts = df['category'].value_counts()
    for cat, count in cat_counts.items():
        percent = (count / total) * 100
        print(f"{cat:<15} : {count:>4} ({percent:.1f}%)")

    # 2. Статистика по интентам
    print("\n--- СТАТИСТИКА ПО ИНТЕНТАМ (intent) ---")
    intent_counts = df['intent'].value_counts()
    for intent, count in intent_counts.items():
        percent = (count / total) * 100
        print(f"{intent:<25} : {count:>4} ({percent:.1f}%)")

    # 3. Анализ дисбаланса
    print("\n--- АНАЛИЗ ДИСБАЛАНСА ---")
    if len(intent_counts) > 0:
        max_intent = intent_counts.index[0]
        max_count = intent_counts.iloc[0]
        min_intent = intent_counts.index[-1]
        min_count = intent_counts.iloc[-1]
        
        print(f"Самый частый класс: {max_intent} ({max_count} шт.)")
        print(f"Самый редкий класс: {min_intent} ({min_count} шт.)")
        print(f"Разница: {max_count - min_count} шт.")
        
        if max_count > min_count * 3:
            print("\n[ВНИМАНИЕ] Сильный дисбаланс! Модель будет переобучаться на частый класс.")
            print("Рекомендуется добавить сообщений для редких классов.")
        else:
            print("\n[ОК] Баланс классов в пределах нормы.")

if __name__ == "__main__":
    check_balance()