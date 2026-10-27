import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error, accuracy_score
import os
from datetime import datetime
import sys
from collections import Counter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА (4 КЛАССА, TF-IDF, БЕЗ MAX_WEIGHT)
# ============================================================
ENABLE_RULES = False
USE_CASE_ID = True
N_SPLITS = 5
BALANCE_MODE = 'custom_smooth'

USE_SVD = False
K_STAR = 73 # Игнорируется, если USE_SVD = False

WEIGHT_BREAK_1 = 0.6
WEIGHT_BREAK_2 = 1.0
# MAX_WEIGHT убран, так как мы отказались от потолка

N_BOOT = 1000
N_BOOT_PER_CLASS = 500
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# --- 1. ЗАГРУЗКА И МАППИНГ ---
corpus_path = os.path.join(DATA_DIR, 'corpus_preprocessed.csv')
if not os.path.exists(corpus_path):
    print("[ОШИБКА] corpus_preprocessed.csv не найден. Запустите этап 2.")
    sys.exit(1)

df = pd.read_csv(corpus_path, sep=';', encoding='utf-8-sig', keep_default_na=False)

# МАППИНГ 16 КЛАССОВ В 4 МАКРОКАТЕГОРИИ
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

def load_int(filename, default=None):
    try:
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            return int(f.read().strip())
    except Exception:
        if default is not None:
            return default
        print(f"[ОШИБКА] Файл {filename} не найден.")
        sys.exit(1)

MIN_DF = load_int('best_min_df.txt')
arch_name = f"LSA (TruncatedSVD, k={K_STAR})" if USE_SVD else "TF-IDF (без SVD)"
cv_mode = "С case_id (честная CV)" if USE_CASE_ID else "БЕЗ case_id (с утечкой)"

print(f"Параметры: min_df* = {MIN_DF}" + (f", k* = {K_STAR}" if USE_SVD else ""))
print(f"Архитектура: {arch_name}")
print(f"Режим CV: {cv_mode} | Фолдов: {N_SPLITS}")
print(f"Балансировка: {BALANCE_MODE} | Градиент: [{WEIGHT_BREAK_1}->{WEIGHT_BREAK_1}, {WEIGHT_BREAK_2}->1.2, без потолка]")

# --- 2. ФИЛЬТРАЦИЯ ---
df_real_all = df[df['is_synthetic'] == 0]
counts = df_real_all['intent'].value_counts()
valid_classes = counts[counts >= 2].index
df_valid = df[df['intent'].isin(valid_classes)].copy()

if 'case_id' not in df_valid.columns:
    USE_CASE_ID = False
    df_valid['case_id'] = 'missing_case'
else:
    df_valid['case_id'] = df_valid['case_id'].fillna('missing_case').astype(str)

df_real = df_valid[df_valid['is_synthetic'] == 0].copy()
df_synth = df_valid[df_valid['is_synthetic'] == 1].copy()

# --- 3. ВСПОМОГАТЕЛЬНЫЕ ---
def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

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

def calc_cwr(y_true, y_pred, weights):
    classes = list(weights.keys())
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    w = np.array([weights[c] for c in classes])
    denom = np.sum(w * (tp + fn))
    return np.sum(w * tp) / denom if denom > 0 else 0.0

def bootstrap_ci_metric(y_true, y_pred, metric_fn, n_boot=N_BOOT, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            continue
        if v is not None and not np.isnan(v):
            values.append(v)
    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))

def urgency_acc_on_correct(y_true_int, y_pred_int, y_true_urg, y_pred_urg):
    mask = (y_true_int == y_pred_int)
    if mask.sum() == 0:
        return np.nan
    return accuracy_score(y_true_urg[mask], y_pred_urg[mask])

def bootstrap_ci_urgency_acc(y_true_int, y_pred_int, y_true_urg, y_pred_urg,
                              n_boot=N_BOOT, alpha=0.05, seed=45):
    rng = np.random.default_rng(seed)
    n = len(y_true_int)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = urgency_acc_on_correct(y_true_int[idx], y_pred_int[idx],
                                   y_true_urg[idx], y_pred_urg[idx])
        if v is not None and not np.isnan(v):
            values.append(v)
    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))

def bootstrap_f1_per_class(y_true, y_pred, labels, n_boot=N_BOOT_PER_CLASS, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.full((n_boot, len(labels)), np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            f1s = f1_score(y_true[idx], y_pred[idx], average=None, labels=labels, zero_division=0)
            values[i] = f1s
        except Exception:
            continue
    lo = np.nanpercentile(values, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(values, 100 * (1 - alpha / 2), axis=0)
    return lo, hi

# --- 4. СПЛИТЫ ---
fold_data = []
if USE_CASE_ID:
    cases_df = df_real.groupby('case_id')['intent'].agg(lambda x: x.mode()[0]).reset_index()
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    split_gen = sgkf.split(cases_df, cases_df['intent'], groups=cases_df['case_id'])
    for train_case_idx, test_case_idx in split_gen:
        train_cases = set(cases_df.iloc[train_case_idx]['case_id'])
        test_cases = set(cases_df.iloc[test_case_idx]['case_id'])
        test_df = df_real[df_real['case_id'].isin(test_cases)].copy()
        train_df = pd.concat([
            df_real[df_real['case_id'].isin(train_cases)],
            df_synth[df_synth['case_id'].isin(train_cases)]
        ]).copy()
        fold_data.append((train_df, test_df))
else:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    split_gen = skf.split(df_real, df_real['intent'])
    for train_idx, test_idx in split_gen:
        test_df = df_real.iloc[test_idx].copy()
        train_df = pd.concat([df_real.iloc[train_idx], df_synth]).copy()
        fold_data.append((train_df, test_df))

oof_true_intent, oof_pred_intent = [], []
oof_true_urgency, oof_pred_urgency_before, oof_pred_urgency_after = [], [], []
fold_results = []

print("\nЗапуск кросс-валидации...")

for fold, (train_df, test_df) in enumerate(fold_data):
    count_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=final_stop_words, use_idf=False, norm=None)
    count_matrix = count_vec.fit_transform(train_df['lemmas'])
    frequencies = count_matrix.sum(axis=0).A1
    feature_names_all = count_vec.get_feature_names_out()
    valid_features = [w for w, f in zip(feature_names_all, frequencies) if f >= MIN_DF or w in keep_words]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=final_stop_words, vocabulary=valid_features)
    X_train = vectorizer.fit_transform(train_df['lemmas'])
    X_test = vectorizer.transform(test_df['lemmas'])

    if USE_SVD:
        actual_k = min(K_STAR, X_train.shape[1] - 1)
        svd = TruncatedSVD(n_components=actual_k, random_state=42)
        X_train_ready = svd.fit_transform(X_train)
        X_test_ready = svd.transform(X_test)
    else:
        X_train_ready = X_train
        X_test_ready = X_test

    if BALANCE_MODE in ['custom', 'custom_smooth']:
        intent_counts = train_df['intent'].value_counts()
        max_count = intent_counts.max()
        cw_intent = {}
        for cls, count in intent_counts.items():
            cwr_w = WEIGHTS.get(cls, 0.1)
            raw_weight = np.sqrt(max_count / count) * cwr_w
            
            # Плавный градиент без верхнего потолка
            if raw_weight < WEIGHT_BREAK_1:
                final_weight = WEIGHT_BREAK_1 + (raw_weight / WEIGHT_BREAK_1) * (WEIGHT_BREAK_2 - WEIGHT_BREAK_1)
            elif raw_weight < WEIGHT_BREAK_2:
                final_weight = WEIGHT_BREAK_2 + ((raw_weight - WEIGHT_BREAK_1) / (WEIGHT_BREAK_2 - WEIGHT_BREAK_1)) * 0.2
            else:
                final_weight = raw_weight
                
            cw_intent[cls] = final_weight
        cw_urgency = 'balanced'
    else:
        cw_intent = BALANCE_MODE
        cw_urgency = BALANCE_MODE

    clf_intent = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw_intent)
    clf_intent.fit(X_train_ready, train_df['intent'])
    y_pred_intent = clf_intent.predict(X_test_ready)

    clf_urgency = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw_urgency)
    clf_urgency.fit(X_train_ready, train_df['urgency'])
    y_pred_urgency_before = clf_urgency.predict(X_test_ready)
    y_pred_urgency_after = y_pred_urgency_before.copy()
    
    if ENABLE_RULES:
        for idx, (pred_intent, pred_urg) in enumerate(zip(y_pred_intent, y_pred_urgency_before)):
            if pred_intent in ['угроза_военная', 'угроза_техногенная']:
                if pred_urg < 4: y_pred_urgency_after[idx] = 4
            elif pred_intent in ['угроза_общая']:
                if pred_urg < 3: y_pred_urgency_after[idx] = 3

    fold_true_int = test_df['intent'].values
    fold_true_urg = test_df['urgency'].values

    f1_int = f1_score(fold_true_int, y_pred_intent, average='macro', zero_division=0)
    f1_urg_before = f1_score(fold_true_urg, y_pred_urgency_before, average='macro', zero_division=0)
    f1_urg_after = f1_score(fold_true_urg, y_pred_urgency_after, average='macro', zero_division=0)
    mae_before = mean_absolute_error(fold_true_urg, y_pred_urgency_before)
    mae_after = mean_absolute_error(fold_true_urg, y_pred_urgency_after)
    uacc = urgency_acc_on_correct(fold_true_int, y_pred_intent, fold_true_urg, y_pred_urgency_after)
    uacc_n = int((fold_true_int == y_pred_intent).sum())
    cwr = calc_cwr(fold_true_int, y_pred_intent, WEIGHTS)

    fold_results.append({
        'fold': fold + 1,
        'n_train': len(train_df),
        'n_synth_train': int((train_df['is_synthetic'] == 1).sum()),
        'n_test': len(test_df),
        'f1_intent': f1_int,
        'f1_urgency_before': f1_urg_before,
        'f1_urgency_after': f1_urg_after,
        'mae_before': mae_before,
        'mae_after': mae_after,
        'urgency_acc': uacc,
        'urgency_acc_n': uacc_n,
        'cwr': cwr,
    })

    oof_true_intent.extend(fold_true_int)
    oof_pred_intent.extend(y_pred_intent)
    oof_true_urgency.extend(fold_true_urg)
    oof_pred_urgency_before.extend(y_pred_urgency_before)
    oof_pred_urgency_after.extend(y_pred_urgency_after)

    print(f"Фолд {fold+1}/{N_SPLITS} | Train: {len(train_df):4d} ({int((train_df['is_synthetic']==1).sum()):4d} synth) | "
          f"Test: {len(test_df):4d} | F1_int={f1_int:.3f} | F1_urg={f1_urg_after:.3f} | "
          f"MAE={mae_after:.3f} | UrgAcc={uacc:.3f} (n={uacc_n}) | CWR={cwr:.3f}")

oof_true_intent = np.array(oof_true_intent)
oof_pred_intent = np.array(oof_pred_intent)
oof_true_urgency = np.array(oof_true_urgency)
oof_pred_urgency_before = np.array(oof_pred_urgency_before)
oof_pred_urgency_after = np.array(oof_pred_urgency_after)

# --- 5. mean ± std ПО ФОЛДАМ ---
def mean_std(key):
    vals = np.array([r[key] for r in fold_results], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

fold_stats = {
    'f1_intent': mean_std('f1_intent'),
    'f1_urgency_before': mean_std('f1_urgency_before'),
    'f1_urgency_after': mean_std('f1_urgency_after'),
    'mae_before': mean_std('mae_before'),
    'mae_after': mean_std('mae_after'),
    'urgency_acc': mean_std('urgency_acc'),
    'cwr': mean_std('cwr'),
}

print("\n" + "="*70)
print("СРЕДНЕЕ ± СТ.ОТКЛ. ПО ФОЛДАМ (CV variance)")
print("="*70)
for key, label in [
    ('f1_intent', 'F1_intent (Macro)'),
    ('f1_urgency_before', 'F1_urgency (до правил)'),
    ('f1_urgency_after', 'F1_urgency (после правил)'),
    ('mae_before', 'MAE_urgency (до правил)'),
    ('mae_after', 'MAE_urgency (после правил)'),
    ('urgency_acc', 'Urgency Acc on Correct Intent'),
    ('cwr', 'CWR'),
]:
    m, s = fold_stats[key]
    print(f"  {label:<35}: {m:.3f} ± {s:.3f}")

# --- 6. АГРЕГАТ OOF ---
f1_int = f1_score(oof_true_intent, oof_pred_intent, average='macro', zero_division=0)
f1_urg_before = f1_score(oof_true_urgency, oof_pred_urgency_before, average='macro', zero_division=0)
f1_urg_after = f1_score(oof_true_urgency, oof_pred_urgency_after, average='macro', zero_division=0)
mae_before = mean_absolute_error(oof_true_urgency, oof_pred_urgency_before)
mae_after = mean_absolute_error(oof_true_urgency, oof_pred_urgency_after)
acc = accuracy_score(oof_true_intent, oof_pred_intent)
cwr = calc_cwr(oof_true_intent, oof_pred_intent, WEIGHTS)
correct_mask = (oof_true_intent == oof_pred_intent)
cond_acc = accuracy_score(oof_true_urgency[correct_mask], oof_pred_urgency_after[correct_mask]) if correct_mask.sum() > 0 else np.nan
cond_acc_n = int(correct_mask.sum())

print("\n" + "="*70)
print(f"РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (4 КЛАССА, TF-IDF)")
print("="*70)
print(f"1. F1_intent (Macro):              {f1_int:.3f}")
print(f"2. F1_urgency (до правил):         {f1_urg_before:.3f}")
print(f"   F1_urgency (после правил):      {f1_urg_after:.3f}")
print(f"   MAE_urgency (до правил):        {mae_before:.3f}")
print(f"   MAE_urgency (после правил):     {mae_after:.3f}")
print(f"3. Urgency Acc on Correct Intent:  {cond_acc:.3f} (n={cond_acc_n})")
print(f"4. Cost-Weighted Recall (CWR):     {cwr:.3f}")
print(f"5. Accuracy:                       {acc:.3f}")

# --- 7. BOOTSTRAP CI ---
print("\n" + "="*70)
print(f"95% ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (bootstrap, {N_BOOT} итераций)")
print("="*70)

ci_f1_intent = bootstrap_ci_metric(
    oof_true_intent, oof_pred_intent,
    lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
    n_boot=N_BOOT, seed=42)
ci_acc = bootstrap_ci_metric(
    oof_true_intent, oof_pred_intent,
    lambda yt, yp: accuracy_score(yt, yp),
    n_boot=N_BOOT, seed=43)
ci_cwr = bootstrap_ci_metric(
    oof_true_intent, oof_pred_intent,
    lambda yt, yp: calc_cwr(yt, yp, WEIGHTS),
    n_boot=N_BOOT, seed=44)
ci_urgency_acc = bootstrap_ci_urgency_acc(
    oof_true_intent, oof_pred_intent, oof_true_urgency, oof_pred_urgency_after,
    n_boot=N_BOOT, seed=45)

print(f"  F1_intent (Macro):  [{ci_f1_intent[0]:.3f}; {ci_f1_intent[1]:.3f}]")
print(f"  Accuracy:           [{ci_acc[0]:.3f}; {ci_acc[1]:.3f}]")
print(f"  Urgency Acc:        [{ci_urgency_acc[0]:.3f}; {ci_urgency_acc[1]:.3f}]")
print(f"  CWR:                [{ci_cwr[0]:.3f}; {ci_cwr[1]:.3f}]")

# --- 8. КЛАССЫ ---
labels_test = sorted(set(oof_true_intent) | set(oof_pred_intent))
cm_intent = confusion_matrix(oof_true_intent, oof_pred_intent, labels=labels_test)
report_intent = classification_report(oof_true_intent, oof_pred_intent, labels=labels_test, zero_division=0)
report_urg_after = classification_report(oof_true_urgency, oof_pred_urgency_after, zero_division=0)
report_urg_before = classification_report(oof_true_urgency, oof_pred_urgency_before, zero_division=0)

lo_per_class, hi_per_class = bootstrap_f1_per_class(
    oof_true_intent, oof_pred_intent, labels_test, n_boot=N_BOOT_PER_CLASS, seed=46)

# --- 9. СОХРАНЕНИЕ ---
val_dir = os.path.join(PLOTS_DIR, 'validation')
os.makedirs(val_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
arch_str = "LSA" if USE_SVD else "TFIDF"
mode_str = f"case_id_{BALANCE_MODE}_{arch_str}_macro4"

# === ГРАФИК №1: Матрица ошибок по интентам ===
plt.figure(figsize=(10, 8)) # Уменьшили размер, так как классов теперь 4
sns.heatmap(cm_intent, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_test, yticklabels=labels_test)
plt.title(f'Матрица ошибок (4 макрокласса) [Out-of-Fold, {N_SPLITS} фолдов]\n'
          f'Архитектура: {arch_str} | Строк: {len(oof_true_intent)}')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
cm_file = os.path.join(val_dir, f'confusion_matrix_intent_{mode_str}_{timestamp}.png')
plt.savefig(cm_file, dpi=300)
plt.close()

# === ГРАФИК №2: Матрица ошибок по urgency ===
urgency_labels = sorted(set(oof_true_urgency) | set(oof_pred_urgency_before) | set(oof_pred_urgency_after))
cm_urg_before = confusion_matrix(oof_true_urgency, oof_pred_urgency_before, labels=urgency_labels)
cm_urg_after = confusion_matrix(oof_true_urgency, oof_pred_urgency_after, labels=urgency_labels)

if ENABLE_RULES:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, cm_data, title in zip(
        axes,
        [cm_urg_before, cm_urg_after],
        ['Urgency: до правил', 'Urgency: после правил']
    ):
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=urgency_labels, yticklabels=urgency_labels, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Предсказанная срочность')
        ax.set_ylabel('Истинная срочность')
    fig.suptitle(f'Матрица ошибок по срочности [Out-of-Fold, {N_SPLITS} фолдов]\n'
                 f'Архитектура: {arch_str} | Строк: {len(oof_true_urgency)}',
                 fontsize=13)
    plt.tight_layout()
    cm_urg_file = os.path.join(val_dir, f'confusion_matrix_urgency_{mode_str}_{timestamp}.png')
    plt.savefig(cm_urg_file, dpi=300)
    plt.close()
else:
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_urg_after, annot=True, fmt='d', cmap='Oranges',
                xticklabels=urgency_labels, yticklabels=urgency_labels)
    plt.title(f'Матрица ошибок по срочности [Out-of-Fold, {N_SPLITS} фолдов]\n'
              f'Архитектура: {arch_str} | Строк: {len(oof_true_urgency)}')
    plt.ylabel('Истинная срочность')
    plt.xlabel('Предсказанная срочность')
    plt.tight_layout()
    cm_urg_file = os.path.join(val_dir, f'confusion_matrix_urgency_{mode_str}_{timestamp}.png')
    plt.savefig(cm_urg_file, dpi=300)
    plt.close()

print(f"Матрица ошибок (intent) сохранена: {cm_file}")
print(f"Матрица ошибок (urgency) сохранена: {cm_urg_file}")

# === ГРАФИК №3: Per-class F1 с 95% CI ===
true_counts = Counter(oof_true_intent)
per_class_rows = []
for i, cls in enumerate(labels_test):
    f1_cls = f1_score(oof_true_intent, oof_pred_intent, labels=[cls], average=None, zero_division=0)[0]
    per_class_rows.append({
        'class': cls,
        'f1': f1_cls,
        'support': true_counts.get(cls, 0),
        'lo': lo_per_class[i],
        'hi': hi_per_class[i],
    })
per_class_rows.sort(key=lambda r: r['f1'])

classes_sorted = [r['class'] for r in per_class_rows]
f1_sorted = np.array([r['f1'] for r in per_class_rows])
lo_sorted = np.array([r['lo'] for r in per_class_rows])
hi_sorted = np.array([r['hi'] for r in per_class_rows])
sup_sorted = np.array([r['support'] for r in per_class_rows])

err_lower = np.clip(f1_sorted - lo_sorted, 0, None)
err_upper = np.clip(hi_sorted - f1_sorted, 0, None)

fig, ax = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(classes_sorted))
colors = plt.cm.RdYlGn(f1_sorted)
bars = ax.barh(y_pos, f1_sorted, xerr=[err_lower, err_upper],
               color=colors, edgecolor='black', capsize=4, alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{c}  (n={s})" for c, s in zip(classes_sorted, sup_sorted)])
ax.set_xlabel('F1-score (Macro по классу)')
ax.set_xlim(0, 1.05)
ax.axvline(f1_int, color='navy', linestyle='--', linewidth=1.5,
           label=f'Общий F1_intent (Macro) = {f1_int:.3f}')
weighted_f1 = f1_score(oof_true_intent, oof_pred_intent, average='weighted', zero_division=0)
ax.axvline(weighted_f1, color='darkgreen', linestyle=':', linewidth=1.5,
           label=f'Weighted F1 = {weighted_f1:.3f}')

for i, (f1_v, s) in enumerate(zip(f1_sorted, sup_sorted)):
    ax.text(min(f1_v + 0.01, 1.02), i, f"{f1_v:.2f}", va='center', fontsize=9)

ax.set_title(f'F1-score по 4 макроклассам с 95% доверительными интервалами\n'
             f'Архитектура: {arch_str} | Out-of-Fold, {N_SPLITS} фолдов',
             fontsize=13, pad=12)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
bar_patch = Patch(facecolor='lightgreen', edgecolor='black', label='F1-score класса')
err_line = Line2D([0], [0], color='black', marker='|', linestyle='none',
                  markersize=10, markeredgewidth=2,
                  label='95% bootstrap CI (усы)')
ax.legend(handles=[bar_patch, err_line, Line2D([0], [0], color='navy', linestyle='--', label='Macro F1'), Line2D([0], [0], color='darkgreen', linestyle=':', label='Weighted F1')], 
          loc='lower right', fontsize=9)

plt.tight_layout()
per_class_file = os.path.join(val_dir, f'per_class_f1_with_ci_{mode_str}_{timestamp}.png')
plt.savefig(per_class_file, dpi=200, bbox_inches='tight')
plt.close()
print(f"Per-class F1 с CI сохранён: {per_class_file}")

# --- 10. ТЕКСТОВЫЙ ОТЧЁТ ---
report_file = os.path.join(val_dir, f'report_{mode_str}_{timestamp}.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(f"Параметры: min_df={MIN_DF}" + (f", k*={K_STAR}" if USE_SVD else "") + "\n")
    f.write(f"Архитектура: {arch_name} (4 макрокласса)\n")
    f.write(f"Режим CV: {cv_mode} | Фолдов: {N_SPLITS}\n")
    f.write(f"Балансировка: {BALANCE_MODE}\n")
    f.write(f"Градиент весов: [{WEIGHT_BREAK_1}->{WEIGHT_BREAK_1}, {WEIGHT_BREAK_2}->1.2, без потолка]\n")
    f.write(f"Правила пост-процессинга (urgency): {'ВКЛ' if ENABLE_RULES else 'ВЫКЛ'}\n")
    f.write(f"Всего строк (Out-of-Fold): {len(oof_true_intent)}\n\n")

    f.write("=== МЕТРИКИ ПО ФОЛДАМ ===\n")
    f.write(f"{'Fold':>4} | {'Train':>6} | {'Synth':>6} | {'Test':>5} | "
            f"{'F1_int':>7} | {'F1_urg':>7} | {'MAE_urg':>8} | {'UrgAcc':>7} | {'CWR':>6}\n")
    f.write("-" * 80 + "\n")
    for r in fold_results:
        uacc_str = f"{r['urgency_acc']:.3f}" if not np.isnan(r['urgency_acc']) else "  n/a"
        f.write(f"{r['fold']:>4} | {r['n_train']:>6} | {r['n_synth_train']:>6} | {r['n_test']:>5} | "
                f"{r['f1_intent']:>7.3f} | {r['f1_urgency_after']:>7.3f} | {r['mae_after']:>8.3f} | "
                f"{uacc_str:>7} | {r['cwr']:>6.3f}\n")
    f.write("\n")

    f.write("=== СРЕДНЕЕ ± СТ.ОТКЛ. ПО ФОЛДАМ ===\n")
    for key, label in [
        ('f1_intent', 'F1_intent (Macro)'),
        ('f1_urgency_before', 'F1_urgency (до правил)'),
        ('f1_urgency_after', 'F1_urgency (после правил)'),
        ('mae_before', 'MAE_urgency (до правил)'),
        ('mae_after', 'MAE_urgency (после правил)'),
        ('urgency_acc', 'Urgency Acc on Correct Intent'),
        ('cwr', 'CWR'),
    ]:
        m, s = fold_stats[key]
        f.write(f"  {label:<35}: {m:.3f} ± {s:.3f}\n")
    f.write("\n")

    f.write("=== СВОДКА МЕТРИК (Out-of-Fold, по всей выборке) ===\n")
    f.write(f"F1_intent (Macro): {f1_int:.3f}\n")
    f.write(f"F1_urgency (до правил): {f1_urg_before:.3f}\n")
    f.write(f"F1_urgency (после правил): {f1_urg_after:.3f}\n")
    f.write(f"MAE_urgency (до правил): {mae_before:.3f}\n")
    f.write(f"MAE_urgency (после правил): {mae_after:.3f}\n")
    f.write(f"Urgency Acc on Correct Intent: {cond_acc:.3f} (на {cond_acc_n} объектах)\n")
    f.write(f"Cost-Weighted Recall (CWR): {cwr:.3f}\n")
    f.write(f"Accuracy: {acc:.3f}\n\n")

    f.write(f"=== 95% ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (bootstrap, {N_BOOT} итераций) ===\n")
    f.write(f"F1_intent (Macro): [{ci_f1_intent[0]:.3f}; {ci_f1_intent[1]:.3f}]\n")
    f.write(f"Accuracy:          [{ci_acc[0]:.3f}; {ci_acc[1]:.3f}]\n")
    f.write(f"Urgency Acc:       [{ci_urgency_acc[0]:.3f}; {ci_urgency_acc[1]:.3f}]\n")
    f.write(f"CWR:               [{ci_cwr[0]:.3f}; {ci_cwr[1]:.3f}]\n\n")

    f.write(f"=== F1 ПО КЛАССАМ + 95% CI (bootstrap, {N_BOOT_PER_CLASS} итераций) ===\n")
    f.write(f"{'Макрокласс':<28} | {'F1':>6} | {'support':>8} | 95% CI\n")
    f.write("-" * 70 + "\n")
    for r in sorted(per_class_rows, key=lambda x: -x['f1']):
        f.write(f"{r['class']:<28} | {r['f1']:>6.3f} | {r['support']:>8d} | [{r['lo']:.3f}; {r['hi']:.3f}]\n")
    f.write("\n")

    f.write("=== МЕТРИКИ МАКРОКЛАССОВ (Out-of-Fold) ===\n")
    f.write(report_intent + "\n\n")
    f.write("=== МЕТРИКИ URGENCY (ПОСЛЕ ПРАВИЛ, Out-of-Fold) ===\n")
    f.write(report_urg_after + "\n")
    if ENABLE_RULES:
        f.write("\n=== МЕТРИКИ URGENCY (ДО ПРАВИЛ, Out-of-Fold) ===\n")
        f.write(report_urg_before + "\n")

print(f"\nТекстовый отчёт сохранён: {report_file}")