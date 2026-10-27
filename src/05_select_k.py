import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

metrics_path = os.path.join(DATA_DIR, 'k_metrics.csv')

if not os.path.exists(metrics_path):
    print(f"[ОШИБКА] Файл {metrics_path} не найден. Сначала запустите этап 4 (04_find_k.py).")
    sys.exit(1)

metrics_df = pd.read_csv(metrics_path, sep=';')
k_range = metrics_df['k'].values
f1_scores = metrics_df['f1_score'].values

# Защита от пустого датафрейма
if len(k_range) == 0:
    print("[ОШИБКА] Файл k_metrics.csv пуст. Этап 4 отработал некорректно.")
    sys.exit(1)

max_f1 = np.nanmax(f1_scores)
best_k_by_f1 = k_range[np.nanargmax(f1_scores)]

print(f"Максимальный F1 = {max_f1:.3f} (при k={best_k_by_f1})")
print("Ищем k* по правилу F1-плато (без учета Reconstruction Error)...\n")

k_star = None

# Шагаем по k, начиная с 10 (отсекаем слишком маленькие размерности)
for i in range(len(k_range)):
    if k_range[i] < 10:
        continue
        
    current_k = k_range[i]
    
    # Правило: смотрим прирост F1 за последние 5 шагов (чтобы сгладить шум)
    lookback = min(5, i)
    past_f1 = f1_scores[i - lookback]
    f1_growth = f1_scores[i] - past_f1
    
    # Если прирост F1 за 5 шагов меньше 1.5%, считаем что вышли на плато
    if f1_growth < 0.015:
        k_star = current_k
        print(f"[+] Найдено начало плато при k = {k_star}")
        print(f"    Прирост F1 за последние {lookback} шагов: {f1_growth:.4f} (меньше 0.015)")
        break

# Если не нашли плато, ищем точку 90% от максимума F1
if k_star is None:
    print("[!] Не удалось найти плато. Ищем точку 90% от максимума F1.")
    target_f1 = max_f1 * 0.90
    valid_indices = np.where(f1_scores >= target_f1)[0]
    
    if len(valid_indices) > 0:
        k_star = k_range[valid_indices[0]]
        print(f"[!] ВНИМАНИЕ: Выбран k*={k_star} по правилу 90% F1.")
    else:
        k_star = best_k_by_f1
        print(f"[!] КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Выбран k*={k_star} (максимум F1).")

idx = np.where(k_range == k_star)[0][0]
f1_at_k = f1_scores[idx]

if max_f1 > 0:
    f1_pct = (f1_at_k / max_f1) * 100
else:
    f1_pct = 0.0

print(f"\n[РЕЗУЛЬТАТ]")
print(f"Выбран k* = {k_star}")
print(f"F1 на этой точке: {f1_at_k:.3f} ({f1_pct:.1f}% от максимума)")

with open(os.path.join(DATA_DIR, 'best_k.txt'), 'w') as f:
    f.write(str(k_star))
print(f"\nk* = {k_star} сохранен в data/best_k.txt")