import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix

# ============================================================
# НАСТРОЙКИ ПО УМОЛЧАНИЮ
# ============================================================
DEFAULT_MIN_DF = 7
DEFAULT_MIN_WEIGHT = 1.0
DEFAULT_MAX_WEIGHT = 2.0
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Grid Search для поиска оптимального k* в LSA.")
    parser.add_argument('--start', type=int, default=45, help='Начальное значение k (по умолчанию 45)')
    parser.add_argument('--end', type=int, default=130, help='Конечное значение k (по умолчанию 130)')
    parser.add_argument('--step', type=int, default=10, help='Шаг перебора k (по умолчанию 10)')
    parser.add_argument('--min_df', type=int, default=DEFAULT_MIN_DF, help=f'Порог min_df (по умолчанию {DEFAULT_MIN_DF})')
    return parser.parse_args()

def main():
    args = parse_args()
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots', 'grid_search')
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"=== ЗАПУСК GRID SEARCH ===")
    print(f"Диапазон k: от {args.start} до {args.end} с шагом {args.step}")
    print(f"Фиксированный min_df: {args.min_df}")
    print(f"Балансировка: custom_smooth [{DEFAULT_MIN_WEIGHT}, {DEFAULT_MAX_WEIGHT}]\n")

    # 1. Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig', keep_default_na=False)

    def load_words(filename):
        with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    stop_words = load_words('stop_words.txt')
    keep_words = load_words('keep_words.txt')
    final_stop_words = [word for word in stop_words if word not in keep_words]

    WEIGHTS = {
        'погибший': 0.8, 'военная_угроза': 1.0, 'техногенная_угроза': 0.95,
        'природная_угроза': 0.9, 'травма': 0.75, 'потерялся': 0.7,
        'заблудился': 0.7, 'животная_угроза': 0.7, 'нет_припасов': 0.7,
        'техническая_неисправность': 0.5, 'статус_группы': 0.2,
        'позиция': 0.2, 'погода': 0.2, 'запрос_маршрута': 0.1,
        'запрос_информации': 0.1, 'проверка_связи': 0.1
    }

    def calc_cwr(y_true, y_pred, weights):
        classes = list(weights.keys())
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        w = np.array([weights[c] for c in classes])
        return np.sum(w * tp) / np.sum(w * (tp + fn)) if np.sum(w * (tp + fn)) > 0 else 0

    # 2. Фильтрация и подготовка фолдов
    df_real = df[df['is_synthetic'] == 0].copy()
    counts = df_real['intent'].value_counts()
    valid_classes = counts[counts >= 2].index
    df_valid = df[df['intent'].isin(valid_classes)].copy()
    df_valid['case_id'] = df_valid['case_id'].fillna('missing').astype(str)
    
    df_real = df_valid[df_valid['is_synthetic'] == 0].copy()
    df_synth = df_valid[df_valid['is_synthetic'] == 1].copy()

    cases_df = df_real.groupby('case_id')['intent'].agg(lambda x: x.mode()[0]).reset_index()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    split_gen = list(sgkf.split(cases_df, cases_df['intent'], groups=cases_df['case_id']))

    fold_data = []
    for train_idx, test_idx in split_gen:
        train_cases = set(cases_df.iloc[train_idx]['case_id'])
        test_cases = set(cases_df.iloc[test_idx]['case_id'])
        
        test_df = df_real[df_real['case_id'].isin(test_cases)].copy()
        train_df = pd.concat([
            df_real[df_real['case_id'].isin(train_cases)],
            df_synth[df_synth['case_id'].isin(train_cases)]
        ]).copy()
        fold_data.append((train_df, test_df))

    # 3. Перебор k
    k_range = list(range(args.start, args.end + 1, args.step))
    results = []

    print("Запуск перебора (это может занять несколько минут)...")
    
    for k in k_range:
        oof_true, oof_pred = [], []
        
        for train_df, test_df in fold_data:
            # Векторизация с защитой keep_words
            count_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=final_stop_words, use_idf=False, norm=None)
            count_matrix = count_vec.fit_transform(train_df['lemmas'])
            frequencies = count_matrix.sum(axis=0).A1
            feature_names = count_vec.get_feature_names_out()
            
            valid_features = [w for w, f in zip(feature_names, frequencies) if f >= args.min_df or w in keep_words]
            
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=final_stop_words, vocabulary=valid_features)
            X_train = vectorizer.fit_transform(train_df['lemmas'])
            X_test = vectorizer.transform(test_df['lemmas'])
            
            # SVD
            actual_k = min(k, X_train.shape[1] - 1)
            svd = TruncatedSVD(n_components=actual_k, random_state=42)
            X_train_lsa = svd.fit_transform(X_train)
            X_test_lsa = svd.transform(X_test)
            
            # Кастомные веса
            intent_counts = train_df['intent'].value_counts()
            max_count = intent_counts.max()
            cw_intent = {}
            for cls, count in intent_counts.items():
                weight = np.sqrt(max_count / count) * WEIGHTS.get(cls, 0.1)
                cw_intent[cls] = max(DEFAULT_MIN_WEIGHT, min(weight, DEFAULT_MAX_WEIGHT))
                
            # Обучение
            clf = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw_intent)
            clf.fit(X_train_lsa, train_df['intent'])
            
            oof_true.extend(test_df['intent'])
            oof_pred.extend(clf.predict(X_test_lsa))
            
        f1 = f1_score(oof_true, oof_pred, average='macro', zero_division=0)
        cwr = calc_cwr(oof_true, oof_pred, WEIGHTS)
        
        results.append({'k': k, 'F1_intent': f1, 'CWR': cwr})
        print(f"k={k:3d} | F1_intent: {f1:.4f} | CWR: {cwr:.4f}")

    # 4. Сохранение результатов в CSV
    res_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = os.path.join(DATA_DIR, f'grid_search_{args.start}-{args.end}_{timestamp}.csv')
    res_df.to_csv(csv_filename, sep=';', index=False)
    print(f"\nТаблица результатов сохранена: {csv_filename}")

    # 5. Построение графика
    print("Построение графика...")
    plt.figure(figsize=(12, 7))
    
    # Ось Y для F1
    ax1 = plt.gca()
    ax1.plot(res_df['k'], res_df['F1_intent'], color='tab:blue', marker='o', linewidth=2, label='Macro F1-Score')
    ax1.set_xlabel('Количество компонент (k)', fontsize=12)
    ax1.set_ylabel('Macro F1-Score', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    # Ось Y для CWR
    ax2 = ax1.twinx()
    ax2.plot(res_df['k'], res_df['CWR'], color='tab:red', marker='s', linestyle='--', linewidth=2, label='Cost-Weighted Recall (CWR)')
    ax2.set_ylabel('Cost-Weighted Recall (CWR)', color='tab:red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Выделение максимальных точек для наглядности
    max_f1_k = res_df.loc[res_df['F1_intent'].idxmax()]
    max_cwr_k = res_df.loc[res_df['CWR'].idxmax()]
    
    ax1.scatter(max_f1_k['k'], max_f1_k['F1_intent'], color='blue', s=150, edgecolors='black', zorder=5)
    ax2.scatter(max_cwr_k['k'], max_cwr_k['CWR'], color='red', s=150, edgecolors='black', zorder=5)

    plt.title(f'Зависимость метрик от размерности k\n(Диапазон: {args.start}-{args.end}, шаг {args.step}, min_df={args.min_df})', fontsize=14)
    
    # Объединение легенд
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plot_filename = os.path.join(PLOTS_DIR, f'grid_search_k_{args.start}-{args.end}_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"График сохранен: {plot_filename}")
    print("\nГотово!")

if __name__ == "__main__":
    main()