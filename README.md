# AirFuzzy-ANFIS: Sistem Cerdas Prediksi Polusi Udara (ISPU Indonesia) Berbasis ANFIS 5-Layer

Aplikasi **Final Project Mata Kuliah Teknologi Logika Fuzzy** (Program Studi Informatika) yang mengimplementasikan **Adaptive Neuro-Fuzzy Inference System (ANFIS) 5-Lapisan** dengan algoritma **Hybrid Learning** (*Ridge Least Squares Estimator* + *Backpropagation dengan Normalisasi Langkah Adaptif*) pada studi kasus prediksi **Indeks Standar Pencemar Udara (ISPU)** mengacu pada **Permen LHK RI No. P.14/MENLHK/SETJEN/KUM.1/7/2020**.

---

## 1. Gambaran Umum & Alasan Pemilihan Metode

Permasalahan polusi udara melibatkan enam parameter pencemar utama (**PM10, PM2.5, SO₂, CO, O₃, NO₂**) yang memiliki rentang satuan berbeda serta transisi ambang batas kualitas udara yang tidak kaku (*fuzzy*).

Pemilihan **ANFIS (Takagi-Sugeno Orde-1)** menggabungkan dua paradigma komputasi:
1. **Sistem Inferensi Fuzzy (Takagi-Sugeno):** Merepresentasikan pengetahuan dalam bentuk fungsi keanggotaan (*Gaussian* / *Generalized Bell*) dan aturan *IF-THEN* yang transparan dan dapat ditelusuri.
2. **Jaringan Saraf Tiruan (Artificial Neural Network):** Mengoptimasi parameter premis (pusat $c$, lebar $\sigma$ / $a$, kemiringan $b$) serta parameter konsekuen ($p_{k,j}, r_k$) secara otomatis dari data historis melalui *Hybrid Learning*.

---

## 2. Struktur Proyek

```text
ANFIS/
├── app.py                   # Entry point aplikasi web GUI Streamlit (6 halaman)
├── requirements.txt         # Daftar pustaka dan dependensi Python
├── README.md                # Dokumentasi proyek, teori, contoh manual, & panduan
├── data/
│   └── ispu_indonesia.csv   # Dataset 1.200 observasi harian polusi udara (5 SPKU)
├── src/
│   ├── __init__.py
│   ├── ispu_standard.py     # Breakpoint resmi Permen LHK 14/2020, klasifikasi, validasi input
│   ├── data_generator.py    # Generator deterministik dataset ispu_indonesia.csv
│   ├── preprocessing.py     # MinMaxScalerCustom, pemetaan kolom CSV, pembersihan data
│   ├── anfis.py             # Mesin 5-Layer ANFIS dari nol (Grid/Scatter, Gaussian/Gbell, Hybrid Learning)
│   ├── evaluation.py        # Metrik regresi (RMSE, MAE, R², MAPE), Confusion Matrix, & baseline
│   ├── manual_example.py    # Contoh perhitungan manual 2-input × 2-MF terverifikasi eksak
│   └── visualization.py     # Seluruh grafik interaktif Plotly (arsitektur, MF, kurva training, dll.)
├── ui/
│   ├── __init__.py
│   ├── styles.py            # Tema CSS, kartu KPI, dan badge status ISPU resmi
│   ├── state.py             # Manajemen session_state dan orkestrasi pipeline pelatihan/inferensi
│   ├── page_home.py         # Halaman 1: Beranda & ringkasan KPI
│   ├── page_data.py         # Halaman 2: Unggah CSV, eksplorasi statistik, & input manual
│   ├── page_training.py     # Halaman 3: Konfigurasi & pelatihan ANFIS 5-Layer
│   ├── page_results.py      # Halaman 4: Hasil prediksi, trace Layer 1–5, & ekspor CSV
│   ├── page_evaluation.py   # Halaman 5: Evaluasi metrik, residual, & komparasi baseline
│   └── page_info.py         # Halaman 6: Diagram arsitektur, teori, contoh manual, & pustaka
└── tests/
    └── test_anfis.py        # Pengujian unit, verifikasi gradien analitik, & smoke test GUI
```

---

## 3. Library Python yang Digunakan

| Library | Peran & Penjelasan Penggunaan di Aplikasi |
| :--- | :--- |
| **NumPy** | Implementasi inti matriks 5-lapisan ANFIS, *Forward Ridge LSE* ($(A^TA + \lambda I)^{-1}A^Ty$), serta perhitungan gradien analitik *Backpropagation*. |
| **Pandas** | Pemuatan dataset CSV, pembersihan data hilang/negatif, penyusunan tabel parameter premis, basis aturan, serta ekspor CSV. |
| **SciPy** | Pendukung komputasi numerik dan statistik bersama NumPy/Scikit-learn. |
| **Scikit-learn** | Pembagian data terstratifikasi (`train_test_split`), inisialisasi cluster *Scatter Partitioning* (`KMeans`), metrik evaluasi, dan model pembanding (`LinearRegression`, `MLPRegressor`, `RandomForestRegressor`). |
| **Plotly** | Pembuatan grafik interaktif: diagram arsitektur 5-layer, kurva fungsi keanggotaan awal vs terlatih, kurva konvergensi training, scatter aktual vs prediksi, residual, confusion matrix, dan komparasi model. |
| **Streamlit** | Pembangunan antarmuka pengguna grafis (GUI) web interaktif multi-halaman lengkap dengan unggah file, validasi input, dan progres pelatihan real-time. |

---

## 4. Penjelasan Tahapan Algoritma ANFIS 5-Layer

Untuk masukan ternormalisasi $\mathbf{x} = [x_1, \dots, x_D]^T \in [0, 1]^D$ dan $R$ aturan fuzzy:

1. **Lapisan 1 — Fuzzifikasi (Node Adaptif):**
   - **Gaussian:** $\mu_{j,m}(x_j) = \exp\left(-\frac{(x_j - c_{j,m})^2}{2\sigma_{j,m}^2}\right)$
   - **Generalized Bell:** $\mu_{j,m}(x_j) = \frac{1}{1 + \left(\frac{(x_j - c_{j,m})^2}{a_{j,m}^2} + \varepsilon\right)^{b_{j,m}}}$
2. **Lapisan 2 — Firing Strength (Node Tetap $\Pi$):**
   $$w_k = \prod_{j=1}^{D} \mu_{k,j}(x_j), \quad k = 1, \dots, R$$
3. **Lapisan 3 — Normalisasi Firing Strength (Node Tetap $N$):**
   $$\bar{w}_k = \frac{w_k}{\sum_{m=1}^{R} w_m}$$
4. **Lapisan 4 — Node Konsekuen Takagi-Sugeno Orde-1 (Node Adaptif):**
   $$O_{4,k} = \bar{w}_k f_k = \bar{w}_k \left(\sum_{j=1}^{D} p_{k,j} x_j + r_k\right)$$
5. **Lapisan 5 — Agregasi Output (Node Tetap $\Sigma$):**
   $$\hat{y} = \sum_{k=1}^{R} \bar{w}_k f_k$$

### Hybrid Learning Algorithm (Setiap Epoch)
- **Forward Pass (Ridge LSE):** Parameter premis tetap, seluruh koefisien konsekuen $\theta$ dihitung sekaligus secara analitik: $\theta^* = (A^T A + \lambda I)^{-1} A^T y$.
- **Backward Pass (Backpropagation):** Parameter konsekuen tetap, gradien fungsi rugi $L = \frac{1}{2N}\sum_{n=1}^{N}(\hat{y}_n - y_n)^2$ diturunkan analitik terhadap $(c, \sigma)$ atau $(c, a, b)$ dan diperbarui dengan langkah ternormalisasi adaptif ala Jang (1993).

---

## 5. Contoh Perhitungan Manual (2 Input × 2 MF = 4 Aturan)

Contoh berikut dapat diverifikasi langsung melalui `src/manual_example.py` dan halaman **6. Informasi Sistem & Teori** di GUI:

- **Input Mentah:** $\text{PM2.5} = 85{,}0\ \mu\text{g/m}^3$ (rentang $[5{,}0, 205{,}0]$) dan $\text{PM10} = 180{,}0\ \mu\text{g/m}^3$ (rentang $[20{,}0, 340{,}0]$).
- **Normalisasi Min-Max:**
  $$x_1 = \frac{85 - 5}{205 - 5} = 0{,}40, \qquad x_2 = \frac{180 - 20}{340 - 20} = 0{,}50$$
- **Parameter Premis Gaussian:** untuk kedua input, MF *Rendah* memiliki $(c = 0{,}20,\ \sigma = 0{,}30)$ dan MF *Tinggi* memiliki $(c = 0{,}80,\ \sigma = 0{,}30)$.
- **Layer 1 (Fuzzifikasi):**
  - $\mu_{\text{PM2.5, Rendah}}(0{,}40) = \exp\left(-\frac{(0{,}40 - 0{,}20)^2}{2 \cdot 0{,}30^2}\right) = 0{,}8007$
  - $\mu_{\text{PM2.5, Tinggi}}(0{,}40) = \exp\left(-\frac{(0{,}40 - 0{,}80)^2}{2 \cdot 0{,}30^2}\right) = 0{,}4111$
  - $\mu_{\text{PM10, Rendah}}(0{,}50) = \exp\left(-\frac{(0{,}50 - 0{,}20)^2}{2 \cdot 0{,}30^2}\right) = 0{,}6065$
  - $\mu_{\text{PM10, Tinggi}}(0{,}50) = \exp\left(-\frac{(0{,}50 - 0{,}80)^2}{2 \cdot 0{,}30^2}\right) = 0{,}6065$
- **Layer 2, 3, & 4 ($\sum w_k = 1{,}4700$):**

| Aturan | Premis | $w_k$ (L2) | $\bar{w}_k$ (L3) | Fungsi Konsekuen $f_k$ | $f_k$ (L4) | $\bar{w}_k \cdot f_k$ |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **R1** | PM2.5 Rendah AND PM10 Rendah | $0{,}4857$ | $0{,}3304$ | $20x_1 + 15x_2 + 25$ | $40{,}5000$ | $13{,}3803$ |
| **R2** | PM2.5 Rendah AND PM10 Tinggi | $0{,}4857$ | $0{,}3304$ | $30x_1 + 95x_2 + 50$ | $109{,}5000$ | $36{,}1764$ |
| **R3** | PM2.5 Tinggi AND PM10 Rendah | $0{,}2494$ | $0{,}1696$ | $110x_1 + 35x_2 + 65$ | $126{,}5000$ | $21{,}4572$ |
| **R4** | PM2.5 Tinggi AND PM10 Tinggi | $0{,}2494$ | $0{,}1696$ | $160x_1 + 140x_2 + 120$ | $254{,}0000$ | $43{,}0839$ |

- **Layer 5 (Agregasi Output):**
  $$\hat{y} = 13{,}3803 + 36{,}1764 + 21{,}4572 + 43{,}0839 = \mathbf{114{,}0978}\ (\text{Kategori: Tidak Sehat})$$

---

## 6. Petunjuk Instalasi & Menjalankan Aplikasi

1. Pastikan **Python 3.10+** sudah terpasang.
2. Instal dependensi proyek:
   ```bash
   pip install -r requirements.txt
   ```
3. *(Opsional)* Buat ulang dataset bawaan `data/ispu_indonesia.csv`:
   ```bash
   python src/data_generator.py
   ```
4. Jalankan aplikasi web GUI Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Jalankan seluruh unit test, verifikasi gradien analitik, dan smoke test GUI:
   ```bash
   python -m unittest discover -s tests -v
   ```

---

## 7. Template Pembagian Tugas & Kontribusi Anggota Kelompok

*(Silakan sesuaikan Nama/NIM sesuai anggota kelompok Anda sebelum pengumpulan laporan resmi)*

| No. | Nama / NIM | Tanggung Jawab | Kontribusi Konkret | Bukti Pekerjaan |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Anggota 1 / NIM | Analisis Masalah & Domain ISPU | Perumusan standar Permen LHK No. 14/2020, generator dataset, BAB I & BAB III | `src/ispu_standard.py`, `src/data_generator.py`, `data/ispu_indonesia.csv` |
| 2 | Anggota 2 / NIM | Pengembangan Algoritma ANFIS | Implementasi mesin 5-Layer ANFIS (Gaussian, Gbell, Grid/Scatter) & Hybrid Learning | `src/anfis.py`, `src/manual_example.py` |
| 3 | Anggota 3 / NIM | Prapemrosesan & Evaluasi | Normalisasi Min-Max, pembersihan CSV unggahan, metrik evaluasi, & komparasi baseline | `src/preprocessing.py`, `src/evaluation.py` |
| 4 | Anggota 4 / NIM | Antarmuka GUI & Visualisasi | Pengembangan GUI multi-halaman Streamlit dan seluruh grafik interaktif Plotly | `app.py`, `ui/*.py`, `src/visualization.py` |
| 5 | Anggota 5 / NIM | Pengujian & Dokumentasi | Pengujian gradien/akurasi/GUI, contoh manual, penyusunan laporan, & slide presentasi | `tests/test_anfis.py`, `README.md`, Laporan & PPT |
