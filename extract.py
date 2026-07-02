# ============================================================
# Import Library
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Scikit-Learn: Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Scikit-Learn: Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Scikit-Learn: Evaluasi
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Konfigurasi tampilan
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
sns.set_palette('deep')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print('✅ Semua library berhasil diimpor.')

# ============================================================
# Memuat Dataset
# ============================================================
df = pd.read_csv('water_potability.csv')

# Menampilkan 5 baris pertama
print('=' * 60)
print('PREVIEW DATA (5 BARIS PERTAMA)')
print('=' * 60)
df.head()

# Dimensi dataset
print(f'Jumlah baris   : {df.shape[0]}')
print(f'Jumlah kolom   : {df.shape[1]}')
print(f'Jumlah fitur   : {df.shape[1] - 1}')
print(f'Kolom target   : Potability')
print(f'\nDaftar kolom   : {list(df.columns)}')

# Informasi tipe data dan non-null count
print('=' * 60)
print('INFORMASI DATASET')
print('=' * 60)
df.info()

# ============================================================
# Statistik Deskriptif
# ============================================================
desc_stats = df.describe().T

# Menambahkan kolom Median secara eksplisit
desc_stats['median'] = df.median(numeric_only=True)

# Mengurutkan kolom agar lebih informatif
desc_stats = desc_stats[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]

print('=' * 60)
print('STATISTIK DESKRIPTIF')
print('=' * 60)
desc_stats

# ============================================================
# Analisis Missing Values
# ============================================================
missing_df = pd.DataFrame({
    'Jumlah Missing': df.isnull().sum(),
    'Persentase (%)': (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values('Jumlah Missing', ascending=False)

print('=' * 60)
print('MISSING VALUES')
print('=' * 60)
print(missing_df)
print(f'\nTotal missing values: {df.isnull().sum().sum()}')
print(f'Total data points   : {df.shape[0] * df.shape[1]}')

# Visualisasi Missing Values dengan Heatmap
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='YlOrRd', ax=ax)
ax.set_title('Heatmap Missing Values', fontsize=14, fontweight='bold')
ax.set_xlabel('Fitur')
ax.set_ylabel('Sampel')
plt.tight_layout()
plt.show()

# ============================================================
# Distribusi Target
# ============================================================
target_counts = df['Potability'].value_counts()
target_pct = df['Potability'].value_counts(normalize=True) * 100

print('Distribusi Kelas Target:')
print(f'  Kelas 0 (Tidak Layak Minum): {target_counts[0]} sampel ({target_pct[0]:.1f}%)')
print(f'  Kelas 1 (Layak Minum)      : {target_counts[1]} sampel ({target_pct[1]:.1f}%)')
print(f'  Rasio                       : 1 : {target_counts[0]/target_counts[1]:.2f}')

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
colors = ['#e74c3c', '#2ecc71']
bars = axes[0].bar(['Tidak Layak (0)', 'Layak Minum (1)'], target_counts.values, color=colors, edgecolor='black', linewidth=0.8)
for bar, count in zip(bars, target_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
                 f'{count}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_title('Distribusi Kelas Target', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Jumlah Sampel')

# Pie chart
axes[1].pie(target_counts.values, labels=['Tidak Layak (0)', 'Layak Minum (1)'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=[0.03, 0.03], shadow=True, textprops={'fontsize': 11})
axes[1].set_title('Proporsi Kelas Target', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# Histogram Distribusi Setiap Fitur
# ============================================================
features = df.columns.drop('Potability')

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(features):
    # Histogram terpisah per kelas
    axes[i].hist(df[df['Potability'] == 0][col].dropna(), bins=30, alpha=0.6,
                 label='Tidak Layak (0)', color='#e74c3c', edgecolor='black', linewidth=0.5)
    axes[i].hist(df[df['Potability'] == 1][col].dropna(), bins=30, alpha=0.6,
                 label='Layak Minum (1)', color='#2ecc71', edgecolor='black', linewidth=0.5)
    axes[i].set_title(f'Distribusi {col}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frekuensi')
    axes[i].legend(fontsize=8)

plt.suptitle('Histogram Distribusi Fitur per Kelas Target', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# Boxplot untuk Deteksi Outliers
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.boxplot(data=df, x='Potability', y=col, hue='Potability', ax=axes[i],
                palette=['#e74c3c', '#2ecc71'], width=0.5, legend=False)
    axes[i].set_title(f'Boxplot {col}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Potability')
    axes[i].set_xticklabels(['Tidak Layak (0)', 'Layak Minum (1)'])

plt.suptitle('Boxplot Fitur per Kelas Target (Deteksi Outliers)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# Kuantifikasi Outliers (Metode IQR)
# ============================================================
print('=' * 60)
print('JUMLAH OUTLIERS PER FITUR (Metode IQR)')
print('=' * 60)

outlier_summary = []
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_summary.append({
        'Fitur': col,
        'Jumlah Outlier': len(outliers),
        'Persentase (%)': round(len(outliers) / df[col].notna().sum() * 100, 2),
        'Batas Bawah': round(lower_bound, 2),
        'Batas Atas': round(upper_bound, 2)
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Jumlah Outlier', ascending=False)
outlier_df.set_index('Fitur', inplace=True)
outlier_df

# ============================================================
# Heatmap Korelasi
# ============================================================
fig, ax = plt.subplots(figsize=(12, 9))

corr_matrix = df.corr()

# Mask untuk segitiga atas
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Koefisien Korelasi'},
            vmin=-1, vmax=1)

ax.set_title('Heatmap Korelasi Pearson Antar Fitur', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# Korelasi setiap fitur terhadap target
print('=' * 60)
print('KORELASI FITUR TERHADAP TARGET (Potability)')
print('=' * 60)

corr_target = df.corr()['Potability'].drop('Potability').sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in corr_target.values]
bars = ax.barh(corr_target.index, corr_target.values, color=colors, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, corr_target.values):
    ax.text(val + (0.002 if val > 0 else -0.002), bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', ha='left' if val > 0 else 'right', va='center', fontsize=10)

ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_title('Korelasi Fitur terhadap Target (Potability)', fontsize=13, fontweight='bold')
ax.set_xlabel('Koefisien Korelasi Pearson')
plt.tight_layout()
plt.show()

# ============================================================
# Ringkasan Temuan EDA
# ============================================================
print('=' * 60)
print('RINGKASAN TEMUAN EDA')
print('=' * 60)

print(f'''
1. UKURAN DATASET:
   - {df.shape[0]} sampel dengan {df.shape[1]-1} fitur numerik.
   - Semua fitur bertipe numerik (float64), tidak ada fitur kategorik.

2. MISSING VALUES:
   - Kolom 'ph'             : {df['ph'].isnull().sum()} missing ({df['ph'].isnull().sum()/len(df)*100:.1f}%)
   - Kolom 'Sulfate'        : {df['Sulfate'].isnull().sum()} missing ({df['Sulfate'].isnull().sum()/len(df)*100:.1f}%)
   - Kolom 'Trihalomethanes': {df['Trihalomethanes'].isnull().sum()} missing ({df['Trihalomethanes'].isnull().sum()/len(df)*100:.1f}%)
   - Kolom lainnya tidak memiliki missing values.

3. DISTRIBUSI TARGET (IMBALANCED):
   - Kelas 0 (Tidak Layak): {target_counts[0]} sampel ({target_pct[0]:.1f}%)
   - Kelas 1 (Layak Minum): {target_counts[1]} sampel ({target_pct[1]:.1f}%)
   - Dataset bersifat IMBALANCED — kelas "tidak layak" lebih dominan.

4. DISTRIBUSI FITUR:
   - Mayoritas fitur berdistribusi mendekati normal (bell-shaped).
   - Distribusi kedua kelas sangat tumpang tindih (overlapping),
     mengindikasikan tidak ada fitur tunggal yang menjadi pembeda kuat.

5. OUTLIERS:
   - Beberapa fitur memiliki outliers (terlihat pada boxplot).
   - Outliers akan ditangani secara implisit oleh standardisasi dan
     pemilihan model yang robust (Random Forest).

6. KORELASI:
   - Korelasi antar fitur sangat rendah (mendekati 0),
     menunjukkan tidak ada multikolinearitas yang signifikan.
   - Korelasi setiap fitur terhadap target juga sangat rendah,
     mengindikasikan bahwa hubungan antara fitur dan target mungkin
     bersifat non-linear atau membutuhkan kombinasi fitur.
''')

# ============================================================
# Handling Missing Values dengan Median
# ============================================================

df_original = df.copy()

cols_with_missing = df.columns[df.isnull().any()].tolist()
print(f'Kolom dengan missing values: {cols_with_missing}\n')

# Menggunakan assignment karena inplace=True deprecated di Pandas 3.x
for col in cols_with_missing:
    median_val = df[col].median()
    missing_count = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print(f'  {col:20s} -> {missing_count} nilai diisi dengan median = {median_val:.4f}')

print(f'\nVerifikasi: Total missing values setelah imputasi = {df.isnull().sum().sum()}')

# ============================================================
# Pemeriksaan Tipe Data & Kebutuhan Encoding
# ============================================================
print('Tipe data setiap kolom:')
print(df.dtypes)

# Cek kolom kategorik
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'\nKolom kategorik ditemukan: {categorical_cols if categorical_cols else "TIDAK ADA"}')
print('\n✅ Semua fitur bertipe numerik. Encoding TIDAK diperlukan.')

# ============================================================
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
    stratify=y  # Menjaga proporsi kelas tetap sama
)

print(f'\n--- Hasil Train-Test Split ---')
print(f'Training set : {X_train.shape[0]} sampel ({X_train.shape[0]/len(X)*100:.0f}%)')
print(f'Test set     : {X_test.shape[0]} sampel ({X_test.shape[0]/len(X)*100:.0f}%)')

print(f'\nDistribusi target pada Training set:')
print(y_train.value_counts().to_string())
print(f'\nDistribusi target pada Test set:')
print(y_test.value_counts().to_string())

# ============================================================
# Standardisasi Fitur (StandardScaler)
# ============================================================
scaler = StandardScaler()

# Fit pada training data, kemudian transform keduanya
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konversi kembali ke DataFrame untuk kemudahan inspeksi
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print('Statistik setelah standardisasi (Training set):')
print(X_train_scaled.describe().loc[['mean', 'std']].round(4))
print('\n✅ Standardisasi berhasil. Mean ≈ 0, Std ≈ 1.')

# ============================================================
# Model 1: Logistic Regression
# ============================================================

assert not np.any(np.isnan(X_train_scaled)), 'X_train masih mengandung NaN!'
assert not np.any(np.isnan(X_test_scaled)), 'X_test masih mengandung NaN!'

lr_model = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

print('Logistic Regression berhasil dilatih.')
print(f'   Jumlah iterasi konvergensi: {lr_model.n_iter_[0]}')

# ============================================================
# Model 2: Random Forest
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=200,       # Jumlah pohon
    max_depth=15,           # Kedalaman maksimum
    min_samples_split=5,    # Min sampel untuk split
    min_samples_leaf=2,     # Min sampel di leaf
    random_state=42,
    n_jobs=-1               # Gunakan semua CPU core
)

# Training (Random Forest tidak memerlukan data yang di-scale,
# namun kita tetap menggunakan data yang sudah di-scale agar konsisten)
rf_model.fit(X_train_scaled, y_train)

# Prediksi
rf_pred = rf_model.predict(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

print('✅ Random Forest berhasil dilatih.')
print(f'   Jumlah pohon: {rf_model.n_estimators}')
print(f'   Jumlah fitur: {rf_model.n_features_in_}')

# ============================================================
# Model 3: Support Vector Machine (SVM)
# ============================================================
svm_model = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)
svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

print('SVM berhasil dilatih.')
print(f'   Kernel         : {svm_model.kernel}')
print(f'   Jumlah support vectors: {svm_model.n_support_.sum()}')

# ============================================================
# Fungsi untuk Menghitung Semua Metrik
# ============================================================
def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }

results = [
    evaluate_model('Logistic Regression', y_test, lr_pred, lr_prob),
    evaluate_model('Random Forest', y_test, rf_pred, rf_prob),
    evaluate_model('SVM (RBF Kernel)', y_test, svm_pred, svm_prob)
]

results_df = pd.DataFrame(results).set_index('Model')

print('=' * 70)
print('PERBANDINGAN METRIK EVALUASI KETIGA MODEL')
print('=' * 70)
results_df

# ============================================================
# Visualisasi Perbandingan Metrik
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

colors = ['#3498db', '#2ecc71', '#e74c3c']
model_names = results_df.index.tolist()

for i, (model_name, color) in enumerate(zip(model_names, colors)):
    values = results_df.loc[model_name, metrics].values
    bars = ax.bar(x + i * width, values, width, label=model_name, color=color,
                  edgecolor='black', linewidth=0.5, alpha=0.85)
    # Tambahkan label nilai
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Metrik Evaluasi', fontsize=12)
ax.set_ylabel('Nilai', fontsize=12)
ax.set_title('Perbandingan Performa Tiga Model Klasifikasi', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Classification Report untuk Setiap Model
# ============================================================
models_info = [
    ('Logistic Regression', lr_pred),
    ('Random Forest', rf_pred),
    ('SVM (RBF Kernel)', svm_pred)
]

target_names = ['Tidak Layak (0)', 'Layak Minum (1)']

for name, pred in models_info:
    print('=' * 60)
    print(f'Classification Report: {name}')
    print('=' * 60)
    print(classification_report(y_test, pred, target_names=target_names))
    print()

# ============================================================
# Confusion Matrix untuk Ketiga Model
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_pred = [
    ('Logistic Regression', lr_pred),
    ('Random Forest', rf_pred),
    ('SVM (RBF Kernel)', svm_pred)
]

for ax, (name, pred) in zip(axes, models_pred):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Tidak Layak (0)', 'Layak Minum (1)'],
                yticklabels=['Tidak Layak (0)', 'Layak Minum (1)'],
                linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'fontweight': 'bold'})
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediksi', fontsize=11)
    ax.set_ylabel('Aktual', fontsize=11)

plt.suptitle('Confusion Matrix - Perbandingan Tiga Model', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()

# ============================================================
# Kurva ROC untuk Ketiga Model
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

models_prob = [
    ('Logistic Regression', lr_prob, '#3498db'),
    ('Random Forest', rf_prob, '#2ecc71'),
    ('SVM (RBF Kernel)', svm_prob, '#e74c3c')
]

for name, prob, color in models_prob:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', color=color, linewidth=2.5)

# Garis diagonal (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier (AUC = 0.5)')

ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
ax.set_title('Kurva ROC - Perbandingan Tiga Model', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.show()

# ============================================================
# Feature Importance dari Random Forest
# ============================================================
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=True)

# Visualisasi
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_imp_df)))
bars = ax.barh(feat_imp_df['Fitur'], feat_imp_df['Importance'],
               color=colors, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, feat_imp_df['Importance']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Importance Score (Gini)', fontsize=12)
ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# Tampilkan ranking
print('\nRanking Feature Importance:')
for rank, (_, row) in enumerate(feat_imp_df.iloc[::-1].iterrows(), 1):
    print(f'  {rank}. {row["Fitur"]:25s} → {row["Importance"]:.4f}')

# ============================================================
# DISKUSI DAN INTERPRETASI HASIL
# ============================================================

best_model_name = results_df['F1-Score'].idxmax()
best_f1 = results_df['F1-Score'].max()
best_auc = results_df.loc[best_model_name, 'ROC-AUC']

print('=' * 70)
print('DISKUSI DAN INTERPRETASI HASIL')
print('=' * 70)

print(f'''
--------------------------------------------------------------------
A. RINGKASAN PERFORMA MODEL
--------------------------------------------------------------------
''')
print(results_df.to_string())

print(f'''

--------------------------------------------------------------------
B. MODEL TERBAIK
--------------------------------------------------------------------

Model terbaik berdasarkan F1-Score: **{best_model_name}**
  - F1-Score : {best_f1:.4f}
  - ROC-AUC  : {best_auc:.4f}

F1-Score dipilih sebagai metrik utama karena dataset bersifat imbalanced.
Accuracy saja tidak cukup karena model bisa mendapatkan accuracy tinggi
hanya dengan memprediksi mayoritas kelas (kelas 0).

--------------------------------------------------------------------
C. ANALISIS PER MODEL
--------------------------------------------------------------------

1. LOGISTIC REGRESSION:
   Kelebihan:
   - Sederhana, cepat, dan mudah diinterpretasikan
   - Memberikan koefisien yang menunjukkan pengaruh tiap fitur
   - Cocok sebagai baseline model
   Kekurangan:
   - Hanya dapat menangkap hubungan linear antara fitur dan target
   - Kurang optimal jika hubungan bersifat non-linear
   - Sensitif terhadap outliers dan multikolinearitas

2. RANDOM FOREST:
   Kelebihan:
   - Dapat menangkap hubungan non-linear dan interaksi antar fitur
   - Robust terhadap outliers dan noise
   - Menyediakan feature importance bawaan
   - Mengurangi overfitting melalui ensemble averaging
   Kekurangan:
   - Lebih sulit diinterpretasikan (black-box)
   - Membutuhkan lebih banyak sumber daya komputasi
   - Cenderung bias terhadap kelas mayoritas pada data imbalanced

3. SVM (RBF KERNEL):
   Kelebihan:
   - Efektif di ruang berdimensi tinggi
   - Kernel RBF mampu menangani batas keputusan non-linear
   - Margin-based: berfokus pada sampel yang sulit (support vectors)
   Kekurangan:
   - Lambat untuk dataset besar
   - Sensitif terhadap pemilihan hyperparameter (C, gamma)
   - Kurang optimal pada data imbalanced tanpa penyesuaian

--------------------------------------------------------------------
D. FAKTOR YANG MEMPENGARUHI PERFORMA
--------------------------------------------------------------------

1. IMBALANCED DATASET:
   Dataset memiliki distribusi kelas yang tidak seimbang
   (kelas 0 lebih banyak dari kelas 1). Hal ini menyebabkan model
   cenderung memprediksi kelas mayoritas, sehingga Recall untuk
   kelas minoritas (kelas 1) menjadi rendah.

2. KORELASI FITUR-TARGET RENDAH:
   Semua fitur memiliki korelasi sangat rendah terhadap target
   (mendekati 0). Ini berarti tidak ada fitur tunggal yang menjadi
   prediktor kuat, sehingga model kesulitan untuk membedakan kelas.

3. DISTRIBUSI FITUR YANG TUMPANG TINDIH:
   Distribusi fitur antara kelas 0 dan 1 sangat mirip (overlapping),
   seperti terlihat pada histogram. Ini membuat batas keputusan
   (decision boundary) menjadi sulit ditentukan.

4. MISSING VALUES:
   Terdapat missing values yang signifikan pada kolom ph, Sulfate,
   dan Trihalomethanes. Imputasi dengan median mungkin tidak selalu
   merepresentasikan nilai sebenarnya.

--------------------------------------------------------------------
E. REKOMENDASI PENINGKATAN
--------------------------------------------------------------------

1. Menangani imbalanced class dengan teknik SMOTE atau class_weight.
2. Melakukan hyperparameter tuning dengan GridSearchCV/RandomizedSearchCV.
3. Mencoba feature engineering untuk membuat fitur baru.
4. Menggunakan metode imputasi yang lebih canggih (KNN Imputer).
5. Menerapkan Cross-Validation untuk evaluasi yang lebih robust.
''')

