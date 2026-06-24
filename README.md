# 🧪 Proyek Machine Learning: Klasifikasi Kualitas Air (Water Potability)

Proyek klasifikasi Machine Learning untuk memprediksi apakah air layak minum (*potable*) atau tidak berdasarkan parameter fisikokimia, menggunakan Python dan Scikit-Learn.

## 📋 Deskripsi Proyek

| Item | Detail |
|---|---|
| **Mata Kuliah** | Machine Learning |
| **Topik** | Klasifikasi |
| **Dataset** | [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability) (Kaggle) |
| **Target** | `Potability` — 1 (Layak Minum) / 0 (Tidak Layak) |
| **Algoritma** | Logistic Regression, Random Forest, SVM |

## 📁 Struktur Proyek

```
ProjectML_Kelompok/
├── README.md                         # Dokumentasi proyek
├── ProjectML_NamaKelompok.ipynb      # Notebook utama (kode & analisis)
├── water_potability.csv              # Dataset
├── venv/                             # Virtual environment (tidak di-push)
└── .gitignore                        # File yang diabaikan Git
```

## 🚀 Setup & Instalasi

### Prasyarat

- **Python 3.10+** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)
- **Terminal** — Kitty, Terminal, Command Prompt, atau PowerShell

---

### 1. Clone Repository

```bash
git clone https://github.com/<username>/ProjectML_Kelompok.git
cd ProjectML_Kelompok
```

> 📌 Ganti `<username>` dengan username GitHub pemilik repository.

---

### 2. Buat Virtual Environment

Virtual environment menjaga dependensi proyek terisolasi dari sistem Python global.

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> ⚠️ **PowerShell:** Jika muncul error terkait *execution policy*, jalankan perintah berikut terlebih dahulu:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Setelah aktif, terminal akan menampilkan `(venv)` di awal baris:

```
(venv) ~/ProjectML_Kelompok$
```

---

### 3. Install Dependensi

Pastikan virtual environment sudah aktif, lalu install semua library yang dibutuhkan:

```bash
pip install jupyter scikit-learn pandas numpy matplotlib seaborn
```

**Library yang diinstall:**

| Library | Fungsi |
|---|---|
| `jupyter` | Menjalankan Jupyter Notebook |
| `scikit-learn` | Pemodelan & evaluasi Machine Learning |
| `pandas` | Manipulasi dan analisis data |
| `numpy` | Komputasi numerik |
| `matplotlib` | Visualisasi data (low-level) |
| `seaborn` | Visualisasi data statistik (high-level) |

---

### 4. Jalankan Jupyter Notebook

```bash
jupyter notebook
```

Jupyter akan otomatis terbuka di browser default. Buka file **`ProjectML_NamaKelompok.ipynb`** dan jalankan sel demi sel dengan **Shift + Enter**.

---

## 📊 Dataset

Dataset `water_potability.csv` berisi **3.276 sampel** dengan **9 fitur** parameter kualitas air:

| Fitur | Deskripsi |
|---|---|
| `ph` | Tingkat keasaman/kebasaan air (0–14) |
| `Hardness` | Kapasitas air mengendapkan sabun (mg/L) |
| `Solids` | Total padatan terlarut / TDS (ppm) |
| `Chloramines` | Jumlah kloramin dalam air (ppm) |
| `Sulfate` | Konsentrasi sulfat terlarut (mg/L) |
| `Conductivity` | Konduktivitas listrik air (μS/cm) |
| `Organic_carbon` | Jumlah karbon organik total (ppm) |
| `Trihalomethanes` | Jumlah trihalometana (μg/L) |
| `Turbidity` | Tingkat kekeruhan air (NTU) |
| **`Potability`** | **Target — 1: Layak Minum, 0: Tidak Layak** |

---

## 🔬 Alur Analisis

Notebook mencakup 5 tahap utama:

1. **Eksplorasi Data (EDA)** — Statistik deskriptif, distribusi, missing values, outliers, korelasi
2. **Preprocessing** — Imputasi missing values (median), standardisasi (StandardScaler), train-test split (80:20)
3. **Pemodelan** — Logistic Regression, Random Forest, SVM (RBF Kernel)
4. **Evaluasi** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, kurva ROC
5. **Diskusi** — Interpretasi hasil, perbandingan model, analisis faktor performa

---

## 🤝 Panduan Kontribusi

1. **Pull** perubahan terbaru sebelum mulai bekerja:
   ```bash
   git pull origin main
   ```

2. Buat **branch** baru untuk perubahanmu:
   ```bash
   git checkout -b fitur/nama-fitur
   ```

3. Setelah selesai, **commit** dan **push**:
   ```bash
   git add .
   git commit -m "Deskripsi perubahan"
   git push origin fitur/nama-fitur
   ```

4. Buat **Pull Request** di GitHub untuk review.

> 💡 **Tips:** Sebelum commit notebook, restart kernel dan **Clear All Outputs** terlebih dahulu (`Kernel → Restart & Clear Output`) agar file `.ipynb` tidak terlalu besar.

---

## ⚠️ Troubleshooting

| Masalah | Solusi |
|---|---|
| `python` tidak dikenali | Pastikan Python sudah terinstall dan ditambahkan ke PATH |
| `ModuleNotFoundError` | Pastikan virtual environment aktif (`source venv/bin/activate`) lalu `pip install` ulang |
| PowerShell execution policy error | Jalankan `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Jupyter tidak terbuka di browser | Salin URL `http://localhost:8888/...` dari terminal dan buka manual di browser |
| Kernel tidak terdeteksi di Jupyter | Jalankan `pip install ipykernel` lalu `python -m ipykernel install --user --name=venv` |
