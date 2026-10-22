import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

metrics_df = pd.read_csv(os.path.join(DATA_DIR, 'k_metrics.csv'), sep=';')
k_range = metrics_df['k'].values
recon_errors = metrics_df['recon_error'].values
silhouettes = metrics_df['silhouette'].values
f1_scores = metrics_df['f1_score'].values

max_f1 = np.max(f1_scores)
min_error = np.min(recon_errors)

print(f"Максимальный F1 = {max_f1:.3f} (при k={k_range[np.argmax(f1_scores)]})")
print("Ищем k* по правилу F1-плато с ограничением ошибки (<= 5%)...\n")

k_star = None

# Шагаем по k, начиная с 10
for i in range(len(k_range)):
    if k_range[i] < 10:
        continue
        
    current_k = k_range[i]
    current_error = recon_errors[i]
    error_increase_pct = ((current_error - min_error) / min_error) * 100
    
    # Правило 1: ошибка не должна превышать минимум более чем на 5%
    if error_increase_pct > 5.0:
        continue # Если ошибка слишком большая, пропускаем этот k
        
    # Правило 2: смотрим прирост F1 за последние 5 шагов (чтобы сгладить шум)
    lookback = min(5, i)
    past_f1 = f1_scores[i - lookback]
    f1_growth = f1_scores[i] - past_f1
    
    # Если прирост F1 за 5 шагов меньше 1.5%, считаем что вышли на плато
    if f1_growth < 0.015:
        k_star = current_k
        print(f"[+] Найдено начало плато при k = {k_star}")
        print(f"    Прирост F1 за последние {lookback} шагов: {f1_growth:.3f} (меньше 0.015)")
        break

# Если не нашли плато при жестком ограничении ошибки, ослабляем его и ищем по 95% F1
if k_star is None:
    print("[!] Не удалось найти плато с ошибкой <= 5%. Ищем точку 90% от максимума F1.")
    target_f1 = max_f1 * 0.90
    valid_indices = np.where(f1_scores >= target_f1)[0]
    if len(valid_indices) > 0:
        k_star = k_range[valid_indices[0]]
    else:
        k_star = k_range[np.argmax(f1_scores)]

idx = np.where(k_range == k_star)[0][0]
f1_at_k = f1_scores[idx]
err_pct = ((recon_errors[idx] - min_error) / min_error) * 100

print(f"\n[РЕЗУЛЬТАТ]")
print(f"Выбран k* = {k_star}")
print(f"F1 на этой точке: {f1_at_k:.3f} ({(f1_at_k/max_f1)*100:.1f}% от максимума)")
print(f"Reconstruction Error: {recon_errors[idx]:.2f} (на {err_pct:.1f}% больше минимума)")

with open(os.path.join(DATA_DIR, 'best_k.txt'), 'w') as f:
    f.write(str(k_star))
print(f"\nk* = {k_star} сохранен в data/best_k.txt")