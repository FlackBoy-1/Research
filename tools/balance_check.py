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
    if 'is_synthetic' in df.columns:
        synthetic_count = df['is_synthetic'].sum()
    else:
        synthetic_count = 0
        
    real_count = total - synthetic_count
    syn_percent = (synthetic_count / total) * 100 if total > 0 else 0

    print("=" * 60)
    print(f"ВСЕГО СООБЩЕНИЙ В БАЗЕ: {total}")
    print(f"  Реальных:  {real_count} ({(real_count/total)*100:.1f}%)")
    print(f"  Синтетики: {synthetic_count} ({syn_percent:.1f}%)")
    print("=" * 60)

    # 1. Статистика по категориям
    print("\n--- СТАТИСТИКА ПО КАТЕГОРИЯМ (category) ---")
    cat_counts = df['category'].value_counts()
    for cat, count in cat_counts.items():
        percent = (count / total) * 100
        print(f"{cat:<15} : {count:>4} ({percent:.1f}%)")

    # 2. Статистика по интентам (с разбивкой на реальные/синтетику)
    print("\n--- СТАТИСТИКА ПО ИНТЕНТАМ (intent) ---")
    print(f"{'Интент':<25} | {'Всего':>5} | {'Реал':>5} | {'Синт':>5} | {'% синт':>7}")
    print("-" * 60)
    
    # Группируем по интенту и считаем сумму синтетики
    intent_stats = df.groupby('intent').agg(
        total=('intent', 'size'),
        synthetic=('is_synthetic', 'sum')
    )
    intent_stats['real'] = intent_stats['total'] - intent_stats['synthetic']
    intent_stats['syn_pct'] = (intent_stats['synthetic'] / intent_stats['total']) * 100
    
    # Сортируем по общему количеству (по убыванию)
    intent_stats = intent_stats.sort_values('total', ascending=False)
    
    for intent, row in intent_stats.iterrows():
        # Выделяем красным флагом те классы, где синтетики больше 30%
        flag = " [!]" if row['syn_pct'] > 30 else ""
        print(f"{intent:<25} | {int(row['total']):>5} | {int(row['real']):>5} | {int(row['synthetic']):>5} | {row['syn_pct']:>6.1f}%{flag}")

    # 3. Анализ дисбаланса
    print("\n--- АНАЛИЗ ДИСБАЛАНСА ---")
    if len(intent_stats) > 0:
        max_intent = intent_stats.index[0]
        max_count = intent_stats.iloc[0]['total']
        min_intent = intent_stats.index[-1]
        min_count = intent_stats.iloc[-1]['total']
        
        print(f"Самый частый класс: {max_intent} ({int(max_count)} шт.)")
        print(f"Самый редкий класс: {min_intent} ({int(min_count)} шт.)")
        print(f"Разница: {int(max_count - min_count)} шт.")
        
        if max_count > min_count * 3:
            print("\n[ВНИМАНИЕ] Сильный дисбаланс! Модель будет переобучаться на частый класс.")
            print("Рекомендуется добавить сообщений для редких классов.")
        else:
            print("\n[ОК] Баланс классов в пределах нормы.")

    # 4. Проверка соотношения 3:1
    print("\n--- ПРОВЕРКА СООТНОШЕНИЯ 3:1 ---")
    max_allowed_syn = real_count // 3
    print(f"Текущее реальных: {real_count}")
    print(f"Максимум синтетики для 3:1: {max_allowed_syn}")
    print(f"Текущее синтетики: {synthetic_count}")
    
    if synthetic_count <= max_allowed_syn:
        print("[ОК] Соотношение 3:1 соблюдено.")
    else:
        print(f"[ВНИМАНИЕ] Превышен лимит синтетики! Нужно добавить {synthetic_count - max_allowed_syn} реальных сообщений или удалить лишнюю синтетику.")

if __name__ == "__main__":
    check_balance()