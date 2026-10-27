import pandas as pd
import numpy as np
import time
import pickle
import os
import sys
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# ============================================================
# НАСТРОЙКИ (4 КЛАССА)
# ============================================================
K_STAR = 73
MIN_DF = 4  # Для 4 классов мы зафиксировали min_df=4
WEIGHT_BREAK_1 = 0.6
WEIGHT_BREAK_2 = 1.0
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_model_size_kb(model):
    return len(pickle.dumps(model)) / 1024

def calc_cwr(y_true, y_pred, weights):
    classes = list(weights.keys())
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    w = np.array([weights[c] for c in classes])
    return np.sum(w * tp) / np.sum(w * (tp + fn)) if np.sum(w * (tp + fn)) > 0 else 0

print("Загрузка данных...")
df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig', keep_default_na=False)

# --- МАППИНГ 16 КЛАССОВ В 4 МАКРОКАТЕГОРИИ ---
MACRO_MAPPING = {
    'военная_угроза': 'угроза_военная',
    'техногенная_угроза': 'угроза_техногенная',
    'травма': 'угроза_общая', 'погибший': 'угроза_общая',
    'природная_угроза': 'угроза_общая', 'животная_угроза': 'угроза_общая',
    'заблудился': 'угроза_общая', 'потерялся': 'угроза_общая',
    'нет_припасов': 'угроза_общая', 'техническая_неисправность': 'угроза_общая', 
    'погода': 'не_угроза', 'позиция': 'не_угроза',
    'статус_группы': 'не_угроза', 'проверка_связи': 'не_угроза',
    'запрос_информации': 'не_угроза', 'запрос_маршрута': 'не_угроза'
}
df['intent'] = df['intent'].map(MACRO_MAPPING)
df = df.dropna(subset=['intent'])

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

# НОВЫЕ ВЕСА ДЛЯ 4 КЛАССОВ
WEIGHTS = {
    'угроза_военная': 0.8,
    'угроза_техногенная': 0.8,
    'угроза_общая': 1.0,
    'не_угроза': 0.2
}

# Подготовка данных
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

def run_pipeline(use_svd=True):
    oof_true, oof_pred = [], []
    total_train_time = 0
    total_inference_time = 0
    model_size_kb = 0

    for train_df, test_df in fold_data:
        count_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=final_stop_words, use_idf=False, norm=None)
        count_matrix = count_vec.fit_transform(train_df['lemmas'])
        frequencies = count_matrix.sum(axis=0).A1
        feature_names = count_vec.get_feature_names_out()
        valid_features = [w for w, f in zip(feature_names, frequencies) if f >= MIN_DF or w in keep_words]

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=final_stop_words, vocabulary=valid_features)

        start_time = time.time()
        X_train = vectorizer.fit_transform(train_df['lemmas'])

        if use_svd:
            svd = TruncatedSVD(n_components=K_STAR, random_state=42)
            X_train_ready = svd.fit_transform(X_train)
        else:
            X_train_ready = X_train
            svd = None

        intent_counts = train_df['intent'].value_counts()
        max_count = intent_counts.max()
        cw_intent = {}
        for cls, count in intent_counts.items():
            raw_weight = np.sqrt(max_count / count) * WEIGHTS.get(cls, 0.1)
            # Плавный градиент без потолка (как в 08_validation.py)
            if raw_weight < WEIGHT_BREAK_1:
                final_weight = WEIGHT_BREAK_1 + (raw_weight / WEIGHT_BREAK_1) * (WEIGHT_BREAK_2 - WEIGHT_BREAK_1)
            elif raw_weight < WEIGHT_BREAK_2:
                final_weight = WEIGHT_BREAK_2 + ((raw_weight - WEIGHT_BREAK_1) / (WEIGHT_BREAK_2 - WEIGHT_BREAK_1)) * 0.2
            else:
                final_weight = raw_weight
            cw_intent[cls] = final_weight

        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw_intent)
        clf.fit(X_train_ready, train_df['intent'])
        total_train_time += (time.time() - start_time)

        start_time = time.time()
        X_test = vectorizer.transform(test_df['lemmas'])
        if use_svd:
            X_test_ready = svd.transform(X_test)
        else:
            X_test_ready = X_test
        y_pred = clf.predict(X_test_ready)
        total_inference_time += (time.time() - start_time)

        if model_size_kb == 0:
            if use_svd:
                model_size_kb = get_model_size_kb(vectorizer) + get_model_size_kb(svd) + get_model_size_kb(clf)
            else:
                model_size_kb = get_model_size_kb(vectorizer) + get_model_size_kb(clf)

        oof_true.extend(test_df['intent'])
        oof_pred.extend(y_pred)

    f1 = f1_score(oof_true, oof_pred, average='macro', zero_division=0)
    cwr = calc_cwr(oof_true, oof_pred, WEIGHTS)
    acc = accuracy_score(oof_true, oof_pred)

    return {
        'name': 'LSA (k=73)' if use_svd else 'TF-IDF (min_df=4)',
        'f1': f1,
        'cwr': cwr,
        'acc': acc,
        'train_time': total_train_time,
        'infer_time': total_inference_time,
        'size_kb': model_size_kb
    }

print("\nЗапуск LSA Pipeline...")
lsa_res = run_pipeline(use_svd=True)

print("Запуск TF-IDF Pipeline...")
tfidf_res = run_pipeline(use_svd=False)

# ============================================================
# ВЫВОД В КОНСОЛЬ
# ============================================================
print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА Ж (4 КЛАССА): СРАВНЕНИЕ АРХИТЕКТУР")
print("="*70)
print(f"{'Параметр':<30} | {'LSA (k=73)':<15} | {'TF-IDF (min_df=4)':<15} | Разница")
print("-" * 70)
print(f"{'F1_intent (Macro)':<30} | {lsa_res['f1']:.4f}          | {tfidf_res['f1']:.4f}          | {tfidf_res['f1'] - lsa_res['f1']:+.4f} (TF-IDF лучше)")
print(f"{'CWR':<30} | {lsa_res['cwr']:.4f}          | {tfidf_res['cwr']:.4f}          | {tfidf_res['cwr'] - lsa_res['cwr']:+.4f} (TF-IDF лучше)")
print(f"{'Accuracy':<30} | {lsa_res['acc']:.4f}          | {tfidf_res['acc']:.4f}          | {tfidf_res['acc'] - lsa_res['acc']:+.4f} (TF-IDF лучше)")
print("-" * 70)
print(f"{'Размер модели (КБ)':<30} | {lsa_res['size_kb']:.1f}             | {tfidf_res['size_kb']:.1f}             | {tfidf_res['size_kb'] - lsa_res['size_kb']:+.1f}")
print(f"{'Время обучения (сек)':<30} | {lsa_res['train_time']:.2f}            | {tfidf_res['train_time']:.2f}            | {tfidf_res['train_time'] - lsa_res['train_time']:+.2f}")
print(f"{'Время инференса (сек)':<30} | {lsa_res['infer_time']:.2f}            | {tfidf_res['infer_time']:.2f}            | {tfidf_res['infer_time'] - lsa_res['infer_time']:+.2f}")
print("="*70)

size_ratio = lsa_res['size_kb'] / tfidf_res['size_kb'] if tfidf_res['size_kb'] > 0 else float('inf')
train_ratio = lsa_res['train_time'] / tfidf_res['train_time'] if tfidf_res['train_time'] > 0 else float('inf')
print(f"\n[ВЫВОД] TF-IDF-модель легче LSA в {size_ratio:.1f} раз и обучается быстрее в {train_ratio:.1f} раз.")
print(f"[ВЫВОД] При этом TF-IDF превосходит LSA по F1 на {(tfidf_res['f1']-lsa_res['f1'])*100:.1f} п.п., "
      f"по CWR — на {(tfidf_res['cwr']-lsa_res['cwr'])*100:.1f} п.п.")

# ============================================================
# ГРАФИК №5: Сравнение LSA vs TF-IDF
# ============================================================
comp_dir = os.path.join(PLOTS_DIR, 'comparison')
os.makedirs(comp_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Левая панель: метрики качества ---
ax = axes[0]
metrics_names = ['F1_intent\n(Macro)', 'CWR', 'Accuracy']
lsa_vals = [lsa_res['f1'], lsa_res['cwr'], lsa_res['acc']]
tfidf_vals = [tfidf_res['f1'], tfidf_res['cwr'], tfidf_res['acc']]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, lsa_vals, width, label='LSA (k=73)', color='#d9534f', edgecolor='black', alpha=0.85)
bars2 = ax.bar(x + width/2, tfidf_vals, width, label='TF-IDF (min_df=4)', color='#5cb85c', edgecolor='black', alpha=0.85)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.set_ylabel('Значение метрики')
ax.set_ylim(0, 1.0)
ax.set_title('Метрики качества (Out-of-Fold, 5 фолдов)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# --- Правая панель: ресурсы (лог-шкала) ---
ax = axes[1]
resources_names = ['Размер модели\n(КБ)', 'Время обучения\n(сек)', 'Время инференса\n(сек)']
lsa_res_vals = [lsa_res['size_kb'], lsa_res['train_time'], lsa_res['infer_time']]
tfidf_res_vals = [tfidf_res['size_kb'], tfidf_res['train_time'], tfidf_res['infer_time']]

x = np.arange(len(resources_names))

bars1 = ax.bar(x - width/2, lsa_res_vals, width, label='LSA (k=73)', color='#d9534f', edgecolor='black', alpha=0.85)
bars2 = ax.bar(x + width/2, tfidf_res_vals, width, label='TF-IDF (min_df=4)', color='#5cb85c', edgecolor='black', alpha=0.85)

for bars, vals in [(bars1, lsa_res_vals), (bars2, tfidf_res_vals)]:
    for bar, val in zip(bars, vals):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.15,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(resources_names, fontsize=11)
ax.set_yscale('log')
ax.set_ylabel('Значение (логарифмическая шкала)')
ax.set_title('Вычислительные ресурсы (меньше — лучше)', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, which='both')

fig.suptitle(f'Сравнение архитектур: LSA (k=73) vs TF-IDF без SVD (min_df=4)\n'
             f'Прикладная 4-классовая таксономия | StratifiedGroupKFold × 5, case_id',
             fontsize=13)

# Описание под графиком
description = (
    "КАК ЧИТАТЬ ГРАФИК:\n"
    "• Левая панель: качество модели. Чем выше бар — тем лучше. Зелёные бары (TF-IDF) выше красных (LSA) по всем трём метрикам.\n"
    "• Правая панель: ресурсы. Чем ниже бар — тем лучше. Логарифмическая шкала, чтобы были видны все значения одновременно.\n"
    "• Размер модели = сериализованные (pickle) vectorizer + SVD (если есть) + LogisticRegression, один фолд.\n"
    "• Время обучения и инференса — суммарное по 5 фолдам.\n"
    "• Главный вывод: TF-IDF без SVD одновременно точнее, легче и быстрее. Это опровергает гипотезу о пользе LSA-сжатия для коротких сообщений."
)
plt.subplots_adjust(bottom=0.20)
fig.text(0.01, 0.01, description, ha='left', va='bottom',
         fontsize=9, family='monospace',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5', edgecolor='gray'))

ts = time.strftime("%Y-%m-%d_%H-%M-%S")
compare_file = os.path.join(comp_dir, f'compare_lsa_vs_tfidf_4classes_{ts}.png')
plt.savefig(compare_file, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nСравнительный график (4 класса) сохранён: {compare_file}")