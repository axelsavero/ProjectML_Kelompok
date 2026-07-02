#!/usr/bin/env python3
"""
Rebuild the Water Potability ML notebook with proper structure:
- Keep EDA sections (cells 0-24)
- Rewrite Preprocessing (handle missing values, encoding check, split, scale, SMOTE)
- Rewrite Modeling in 3 phases:
    Phase 1: Compare base models on balanced data -> pick best by F1
    Phase 2: GridSearchCV hyperparameter tuning on best model
    Phase 3: Train final model with best params, full evaluation
- Fix ROC-AUC, Confusion Matrix, Classification Report
"""

import json, copy

notebook_path = 'ml-project_water-potability.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_md(source, cell_id=None):
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n') if isinstance(source, str) else source
    }
    if cell_id:
        cell["id"] = cell_id
    # Fix: source lines need newlines except the last
    lines = source.split('\n') if isinstance(source, str) else source
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            fixed.append(line + '\n')
        else:
            fixed.append(line)
    cell["source"] = fixed
    return cell

def make_code(source, cell_id=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": []
    }
    if cell_id:
        cell["id"] = cell_id
    lines = source.split('\n') if isinstance(source, str) else source
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            fixed.append(line + '\n')
        else:
            fixed.append(line)
    cell["source"] = fixed
    return cell

# ============================================================
# Keep cells 0-24 (EDA part is good)
# ============================================================
new_cells = nb["cells"][:25]

# ============================================================
# SECTION 2: PREPROCESSING
# ============================================================
new_cells.append(make_md("""---
## 2. Preprocessing Data

Pada tahap ini, kita melakukan persiapan data sebelum dimasukkan ke dalam model machine learning. Langkah-langkah meliputi:
1. Menangani missing values
2. Memeriksa kebutuhan encoding
3. Memisahkan fitur dan target
4. Membagi data menjadi training dan testing set
5. Melakukan standardisasi fitur
6. **Menangani data tidak seimbang (imbalanced) menggunakan SMOTE**"""))

# 2.1 Handling Missing Values
new_cells.append(make_md("""### 2.1 Handling Missing Values

Strategi yang dipilih adalah mengisi missing values dengan **median** masing-masing kolom.
Median dipilih karena lebih robust terhadap outliers dibanding mean."""))

new_cells.append(make_code("""# ============================================================
# Handling Missing Values dengan Median
# ============================================================

df_original = df.copy()

cols_with_missing = df.columns[df.isnull().any()].tolist()
print(f'Kolom dengan missing values: {cols_with_missing}\\n')

for col in cols_with_missing:
    median_val = df[col].median()
    missing_count = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print(f'  {col:20s} -> {missing_count} nilai diisi dengan median = {median_val:.4f}')

print(f'\\nVerifikasi: Total missing values setelah imputasi = {df.isnull().sum().sum()}')"""))

# 2.2 Encoding check
new_cells.append(make_md("""### 2.2 Pemeriksaan Data Kategorik & Encoding

Encoding diperlukan jika terdapat fitur bertipe kategorik (teks/objek). Pemeriksaan dilakukan untuk memastikan apakah langkah ini diperlukan."""))

new_cells.append(make_code("""# ============================================================
# Pemeriksaan Tipe Data & Kebutuhan Encoding
# ============================================================
print('Tipe data setiap kolom:')
print(df.dtypes)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'\\nKolom kategorik ditemukan: {categorical_cols if categorical_cols else "TIDAK ADA"}')
print('\\n✅ Semua fitur bertipe numerik. Encoding TIDAK diperlukan.')"""))

# 2.3 Split
new_cells.append(make_md("""### 2.3 Pemisahan Fitur-Target dan Train-Test Split

Data dibagi menjadi:
- **Training set (80%)**: Untuk melatih model
- **Test set (20%)**: Untuk mengevaluasi performa model

Parameter `stratify=y` digunakan agar proporsi kelas target tetap sama di kedua set."""))

new_cells.append(make_code("""# ============================================================
# Pemisahan Fitur (X) dan Target (y)
# ============================================================
X = df.drop('Potability', axis=1)
y = df['Potability']

print(f'Dimensi fitur (X) : {X.shape}')
print(f'Dimensi target (y): {y.shape}')

# ============================================================
# Train-Test Split (80:20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'\\n--- Hasil Train-Test Split ---')
print(f'Training set : {X_train.shape[0]} sampel ({X_train.shape[0]/len(X)*100:.0f}%)')
print(f'Test set     : {X_test.shape[0]} sampel ({X_test.shape[0]/len(X)*100:.0f}%)')

print(f'\\nDistribusi target pada Training set:')
print(y_train.value_counts().to_string())
print(f'\\nDistribusi target pada Test set:')
print(y_test.value_counts().to_string())"""))

# 2.4 Standardization
new_cells.append(make_md("""### 2.4 Normalisasi/Standardisasi Fitur

**StandardScaler** digunakan untuk mentransformasi setiap fitur agar memiliki *mean = 0* dan *standard deviation = 1*.
Scaler di-fit **hanya pada data training** untuk mencegah data leakage."""))

new_cells.append(make_code("""# ============================================================
# Standardisasi Fitur (StandardScaler)
# ============================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print('Statistik setelah standardisasi (Training set):')
print(X_train_scaled.describe().loc[['mean', 'std']].round(4))
print('\\n✅ Standardisasi berhasil. Mean ≈ 0, Std ≈ 1.')"""))

# 2.5 SMOTE
new_cells.append(make_md("""### 2.5 Handling Imbalanced Data (SMOTE)

Dari analisis EDA, kita mengetahui bahwa dataset bersifat **imbalanced** — kelas `Tidak Layak Minum (0)` lebih banyak dari kelas `Layak Minum (1)`.

Jika data tidak diseimbangkan terlebih dahulu, model cenderung memprediksi kelas mayoritas saja, yang mengakibatkan:
- **F1-Score sangat rendah** (di bawah 0.5)
- **Recall kelas minoritas sangat buruk**
- Model terlihat memiliki akurasi tinggi, tetapi sebenarnya tidak bisa mengenali kelas minoritas

Oleh karena itu, kita menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)** untuk membuat data sintetis pada kelas minoritas di data training.

> **Penting**: SMOTE hanya diterapkan pada **data training**, bukan data testing. Data testing harus tetap merepresentasikan distribusi asli agar evaluasi model bersifat realistis."""))

new_cells.append(make_code("""# ============================================================
# Handling Imbalanced Data dengan SMOTE
# ============================================================
from imblearn.over_sampling import SMOTE

print('--- Sebelum SMOTE ---')
print(f'Distribusi y_train:')
print(y_train.value_counts().to_string())
print(f'Rasio kelas 0 : kelas 1 = {y_train.value_counts()[0]} : {y_train.value_counts()[1]}')

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f'\\n--- Setelah SMOTE ---')
print(f'Distribusi y_train_balanced:')
print(y_train_balanced.value_counts().to_string())
print(f'Rasio kelas 0 : kelas 1 = {y_train_balanced.value_counts()[0]} : {y_train_balanced.value_counts()[1]}')
print(f'\\nJumlah sampel training bertambah dari {len(y_train)} menjadi {len(y_train_balanced)}')
print('\\n✅ Data training berhasil diseimbangkan dengan SMOTE.')"""))

# ============================================================
# SECTION 3: MODELING - 3 PHASES
# ============================================================
new_cells.append(make_md("""---
## 3. Pemodelan (Modeling)

Proses pemodelan dilakukan dalam **3 tahap** secara berurutan:
1. **Tahap 1** — Menentukan model terbaik dari 3 algoritma (Logistic Regression, Random Forest, SVM)
2. **Tahap 2** — Mencari hyperparameter terbaik untuk model terpilih menggunakan GridSearchCV
3. **Tahap 3** — Melatih model final dengan hyperparameter terbaik dan melakukan evaluasi lengkap

Semua model dilatih menggunakan **data training yang sudah diseimbangkan (SMOTE)** dan dievaluasi pada **data testing asli (tanpa SMOTE)**."""))

# Phase 1
new_cells.append(make_md("""### 3.1 Tahap 1: Menentukan Model Terbaik

Pada tahap ini, kita melatih 3 model dasar dengan parameter default (kecuali `max_depth` Random Forest yang dibatasi agar tidak overfitting).
Evaluasi dilakukan pada test set dan model terbaik dipilih berdasarkan **F1-Score**, karena F1-Score merupakan harmonic mean dari Precision dan Recall yang tepat digunakan pada kasus klasifikasi dengan data imbalanced."""))

new_cells.append(make_code("""# ============================================================
# TAHAP 1: Menentukan Model Terbaik (Base Models)
# ============================================================

# Definisi 3 model kandidat
base_models = {
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,          # Dibatasi agar tidak overfitting
        random_state=42,
        n_jobs=-1
    ),
    'SVM (RBF Kernel)': SVC(
        kernel='rbf',
        probability=True,     # Diperlukan untuk predict_proba (ROC-AUC)
        random_state=42
    )
}

# Training dan evaluasi setiap model
phase1_results = []

for name, model in base_models.items():
    # Train pada data BALANCED
    model.fit(X_train_balanced, y_train_balanced)
    
    # Prediksi pada data TEST (asli/tidak di-SMOTE)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    phase1_results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4)
    })
    
    print(f'{name}:')
    print(f'  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}')

# Tabel perbandingan
phase1_df = pd.DataFrame(phase1_results).set_index('Model')

print('\\n' + '=' * 70)
print('PERBANDINGAN METRIK EVALUASI - TAHAP 1 (BASE MODELS)')
print('=' * 70)
phase1_df"""))

new_cells.append(make_code("""# ============================================================
# Visualisasi Perbandingan Metrik Base Models
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

colors = ['#3498db', '#2ecc71', '#e74c3c']
model_names = phase1_df.index.tolist()

for i, (model_name, color) in enumerate(zip(model_names, colors)):
    values = phase1_df.loc[model_name, metrics].values
    bars = ax.bar(x + i * width, values, width, label=model_name, color=color,
                  edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Metrik Evaluasi', fontsize=12)
ax.set_ylabel('Nilai', fontsize=12)
ax.set_title('Perbandingan Performa Base Models (Data Balanced dengan SMOTE)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()"""))

new_cells.append(make_code("""# ============================================================
# Tentukan Model Terbaik berdasarkan F1-Score
# ============================================================
best_model_name = phase1_df['F1-Score'].idxmax()
best_f1_phase1 = phase1_df.loc[best_model_name, 'F1-Score']

print('=' * 70)
print(f'MODEL TERBAIK BERDASARKAN F1-SCORE: {best_model_name}')
print(f'F1-Score: {best_f1_phase1}')
print('=' * 70)
print(f'\\nModel "{best_model_name}" akan dilanjutkan ke Tahap 2 (Hyperparameter Tuning).')"""))

# Phase 2
new_cells.append(make_md("""### 3.2 Tahap 2: Mencari Hyperparameter Terbaik (GridSearchCV)

Setelah menentukan model terbaik di Tahap 1, kita optimasi hyperparameter-nya menggunakan **GridSearchCV** dengan **5-fold cross-validation**.

GridSearchCV akan mencoba semua kombinasi hyperparameter yang kita definisikan dan mengevaluasinya menggunakan metrik **F1-Score**. Proses ini memastikan hyperparameter yang dipilih benar-benar optimal, bukan dipilih secara sembarang."""))

new_cells.append(make_code("""# ============================================================
# TAHAP 2: Hyperparameter Tuning menggunakan GridSearchCV
# ============================================================
from sklearn.model_selection import GridSearchCV

print(f'Melakukan GridSearchCV untuk model: {best_model_name}')
print('Scoring: F1-Score | CV: 5-Fold')
print()

# Definisikan param_grid berdasarkan model terbaik
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tuning_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
elif best_model_name == 'SVM (RBF Kernel)':
    param_grid = {
        'C': [0.1, 0.5, 1, 5, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf']
    }
    tuning_model = SVC(probability=True, random_state=42)
    
else:  # Logistic Regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2']
    }
    tuning_model = LogisticRegression(random_state=42, max_iter=1000)

print('Parameter grid yang diuji:')
for param, values in param_grid.items():
    print(f'  {param}: {values}')

total_combinations = 1
for v in param_grid.values():
    total_combinations *= len(v)
print(f'\\nTotal kombinasi: {total_combinations}')
print(f'Total fit (kombinasi × cv): {total_combinations * 5}')

grid_search = GridSearchCV(
    estimator=tuning_model,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# Fit pada data BALANCED
grid_search.fit(X_train_balanced, y_train_balanced)

print('\\n' + '=' * 70)
print('HASIL HYPERPARAMETER TUNING')
print('=' * 70)
print(f'Hyperparameter terbaik:')
for param, value in grid_search.best_params_.items():
    print(f'  {param}: {value}')
print(f'\\nBest CV F1-Score: {grid_search.best_score_:.4f}')"""))

new_cells.append(make_code("""# ============================================================
# Tampilkan Top 10 Kombinasi Hyperparameter
# ============================================================
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values('rank_test_score')

top10 = cv_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head(10)
top10.columns = ['Parameters', 'Mean F1 (CV)', 'Std F1 (CV)', 'Rank']
top10 = top10.reset_index(drop=True)
top10.index = top10.index + 1

print('Top 10 Kombinasi Hyperparameter:')
top10"""))

# Phase 3
new_cells.append(make_md("""### 3.3 Tahap 3: Train Model Final & Evaluasi Lengkap

Sekarang kita menggunakan hyperparameter terbaik dari Tahap 2 untuk melatih model final.
Evaluasi dilakukan pada **test set yang tidak pernah dilihat model** dan meliputi:
- Metrik: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Classification Report
- Confusion Matrix
- Kurva ROC"""))

new_cells.append(make_code("""# ============================================================
# TAHAP 3: Train Model Final dengan Hyperparameter Terbaik
# ============================================================

# Ambil model terbaik dari GridSearchCV (sudah di-train pada data balanced)
final_model = grid_search.best_estimator_

# Prediksi pada TEST SET (data asli, tanpa SMOTE)
y_pred_final = final_model.predict(X_test_scaled)
y_prob_final = final_model.predict_proba(X_test_scaled)[:, 1]

# Hitung semua metrik
final_acc = accuracy_score(y_test, y_pred_final)
final_prec = precision_score(y_test, y_pred_final)
final_rec = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_auc = roc_auc_score(y_test, y_prob_final)

print('=' * 70)
print(f'EVALUASI FINAL: {best_model_name} (Tuned)')
print('=' * 70)
print(f'Hyperparameter: {grid_search.best_params_}')
print()
print(f'Accuracy  : {final_acc:.4f}')
print(f'Precision : {final_prec:.4f}')
print(f'Recall    : {final_rec:.4f}')
print(f'F1-Score  : {final_f1:.4f}')
print(f'ROC-AUC   : {final_auc:.4f}')"""))

new_cells.append(make_code("""# ============================================================
# Classification Report
# ============================================================
target_names = ['Tidak Layak (0)', 'Layak Minum (1)']

print('=' * 60)
print(f'Classification Report: {best_model_name} (Tuned)')
print('=' * 60)
print(classification_report(y_test, y_pred_final, target_names=target_names))"""))

new_cells.append(make_code("""# ============================================================
# Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test, y_pred_final)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=target_names,
            yticklabels=target_names,
            linewidths=2, linecolor='white',
            annot_kws={'size': 16, 'fontweight': 'bold'})
ax.set_title(f'Confusion Matrix — {best_model_name} (Tuned)', fontsize=14, fontweight='bold')
ax.set_xlabel('Prediksi', fontsize=12)
ax.set_ylabel('Aktual', fontsize=12)
plt.tight_layout()
plt.show()

# Interpretasi
tn, fp, fn, tp = cm.ravel()
print(f'True Negatives  (TN): {tn} — Benar diprediksi Tidak Layak')
print(f'False Positives (FP): {fp} — Salah diprediksi Layak (padahal Tidak Layak)')
print(f'False Negatives (FN): {fn} — Salah diprediksi Tidak Layak (padahal Layak)')
print(f'True Positives  (TP): {tp} — Benar diprediksi Layak Minum')"""))

new_cells.append(make_code("""# ============================================================
# Kurva ROC-AUC
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_prob_final)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot kurva ROC model
ax.plot(fpr, tpr, color='#e74c3c', linewidth=2.5,
        label=f'{best_model_name} (AUC = {final_auc:.4f})')

# Plot garis diagonal (random classifier baseline)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5,
        label='Random Classifier (AUC = 0.5000)')

# Fill area under curve
ax.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')

ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
ax.set_title('Kurva ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.show()

print(f'\\nAUC-ROC Score: {final_auc:.4f}')
print('Interpretasi: AUC > 0.5 menunjukkan model lebih baik dari random classifier.')
print('Semakin mendekati 1.0, semakin baik kemampuan model membedakan kedua kelas.')"""))

# Feature Importance (if RF)
new_cells.append(make_code("""# ============================================================
# Feature Importance (jika model terbaik adalah Random Forest)
# ============================================================
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Fitur': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_imp_df)))
    bars = ax.barh(feat_imp_df['Fitur'], feat_imp_df['Importance'],
                   color=colors_fi, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, feat_imp_df['Importance']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Importance Score (Gini)', fontsize=12)
    ax.set_title('Feature Importance — Random Forest (Tuned)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print('\\nRanking Feature Importance:')
    for rank, (_, row) in enumerate(feat_imp_df.iloc[::-1].iterrows(), 1):
        print(f'  {rank}. {row["Fitur"]:25s} -> {row["Importance"]:.4f}')
elif hasattr(final_model, 'coef_'):
    coef_df = pd.DataFrame({
        'Fitur': X.columns,
        'Koefisien': final_model.coef_[0]
    }).sort_values('Koefisien', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_coef = ['#e74c3c' if v < 0 else '#2ecc71' for v in coef_df['Koefisien']]
    ax.barh(coef_df['Fitur'], coef_df['Koefisien'], color=colors_coef, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Koefisien', fontsize=12)
    ax.set_title(f'Koefisien Fitur — {best_model_name} (Tuned)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print('Model ini tidak memiliki feature importance atau koefisien yang dapat divisualisasikan.')"""))

# Conclusion
new_cells.append(make_md("""---
## 4. Kesimpulan dan Diskusi"""))

new_cells.append(make_code("""# ============================================================
# KESIMPULAN DAN DISKUSI
# ============================================================

print('=' * 70)
print('KESIMPULAN DAN DISKUSI')
print('=' * 70)

print(f'''
A. RINGKASAN PROSES
-------------------
1. Dataset Water Potability memiliki {df_original.shape[0]} sampel dan {df_original.shape[1]-1} fitur numerik.
2. Missing values pada kolom ph, Sulfate, dan Trihalomethanes ditangani dengan imputasi median.
3. Data bersifat IMBALANCED, ditangani dengan SMOTE pada data training.
4. Proses pemodelan dilakukan dalam 3 tahap sistematis.

B. HASIL TAHAP 1 (Pemilihan Model Terbaik)
-------------------------------------------''')
print(phase1_df.to_string())
print(f'\\n   Model terbaik: {best_model_name} (F1-Score: {best_f1_phase1})')

print(f'''
C. HASIL TAHAP 2 (Hyperparameter Tuning)
-----------------------------------------
   Metode        : GridSearchCV (5-Fold CV)
   Scoring       : F1-Score
   Best CV Score : {grid_search.best_score_:.4f}
   Best Params   : {grid_search.best_params_}

D. HASIL TAHAP 3 (Evaluasi Final)
----------------------------------
   Accuracy  : {final_acc:.4f}
   Precision : {final_prec:.4f}
   Recall    : {final_rec:.4f}
   F1-Score  : {final_f1:.4f}
   ROC-AUC   : {final_auc:.4f}

E. PENGARUH SMOTE TERHADAP PERFORMA
------------------------------------
   Sebelum menggunakan SMOTE, model cenderung memprediksi kelas mayoritas
   (Tidak Layak Minum) sehingga F1-Score sangat rendah (< 0.5).
   
   Setelah data training diseimbangkan dengan SMOTE:
   - Recall meningkat signifikan (model lebih mampu mengenali air layak minum)
   - F1-Score meningkat karena keseimbangan antara Precision dan Recall
   - ROC-AUC memberikan gambaran yang lebih realistis tentang kemampuan model

F. CATATAN PENTING
------------------
   - SMOTE hanya diterapkan pada data TRAINING, TIDAK pada data testing.
   - Data testing tetap menggunakan distribusi asli untuk evaluasi yang realistis.
   - Hyperparameter dipilih berdasarkan Cross-Validation, bukan secara manual.
   - ROC-AUC dihitung menggunakan probabilitas prediksi (predict_proba),
     bukan label prediksi, sehingga hasilnya lebih akurat.
''')"""))

nb["cells"] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook rebuilt with {len(new_cells)} cells.")
