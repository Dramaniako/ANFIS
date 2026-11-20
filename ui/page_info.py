"""Halaman 6: Informasi Sistem, Landasan Teori ANFIS, Contoh Manual, dan Pustaka."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.manual_example import compute_worked_manual_example
from src.visualization import plot_anfis_architecture


def render_page_info() -> None:
    """Merender dokumentasi metode, arsitektur 5-layer, contoh perhitungan manual, dan panduan."""
    st.title("Informasi Sistem, Teori ANFIS & Perhitungan Manual")
    st.caption(
        "Dokumentasi ilmiah arsitektur ANFIS 5-Lapisan, Algoritma Hybrid Learning, "
        "contoh perhitungan manual terverifikasi, serta daftar pustaka Python."
    )

    model = st.session_state["anfis_model"]
    summary = model.summary()

    st.subheader("1. Diagram Arsitektur ANFIS 5-Lapisan Aktif")
    st.plotly_chart(
        plot_anfis_architecture(
            n_inputs=model.n_inputs,
            feature_names=model.feature_names,
            partition_type=model.partition_type,
            n_mfs=summary["n_mfs"],
            n_rules=model.n_rules,
        ),
        width="stretch",
    )

    tab_theory, tab_manual, tab_guide = st.tabs(
        ["Arsitektur 5-Layer & Hybrid Learning", "Contoh Perhitungan Manual (2 Input × 2 MF)", "Library & Petunjuk Penggunaan"]
    )

    with tab_theory:
        st.markdown(
            """
            ### Arsitektur Jaringan Adaptif ANFIS (Takagi-Sugeno Orde-1)
            ANFIS memetakan vektor masukan ternormalisasi $\\mathbf{x} = [x_1, x_2, \\dots, x_D]^T$ menjadi skor prediksi ISPU $\\hat{y}$ melalui lima lapisan komputasi:

            1. **Lapisan 1 — Fuzzifikasi (Node Adaptif):**
               Setiap simpul menghitung derajat keanggotaan $\\mu_{j,m}(x_j) \\in [0, 1]$ menggunakan kurva **Gaussian** atau **Generalized Bell**:
               $$\\mu_{\\text{Gaussian}}(x_j; c, \\sigma) = \\exp\\left(-\\frac{(x_j - c)^2}{2\\sigma^2}\\right), \\qquad \\mu_{\\text{Gbell}}(x_j; a, b, c) = \\frac{1}{1 + \\left(\\frac{(x_j - c)^2}{a^2} + \\varepsilon\\right)^b}$$
               Parameter $(c, \\sigma)$ atau $(a, b, c)$ disebut **parameter premis** (*non-linear parameters*).

            2. **Lapisan 2 — Firing Strength Aturan (Node Tetap $\\Pi$):**
               Mengalikan seluruh derajat keanggotaan pada bagian premis aturan ke-$k$ (operator *T-norm Product*):
               $$w_k = \\prod_{j=1}^{D} \\mu_{k,j}(x_j), \\quad k = 1, \\dots, R$$

            3. **Lapisan 3 — Normalisasi Firing Strength (Node Tetap $N$):**
               Menghitung rasio kontribusi aturan ke-$k$ terhadap jumlah kekuatan seluruh aturan:
               $$\\bar{w}_k = \\frac{w_k}{\\sum_{m=1}^{R} w_m}$$

            4. **Lapisan 4 — Fungsi Konsekuen Takagi-Sugeno (Node Adaptif):**
               Mengalikan bobot ternormalisasi $\\bar{w}_k$ dengan persamaan linear konsekuen orde-1:
               $$O_{4,k} = \\bar{w}_k f_k = \\bar{w}_k \\left(\\sum_{j=1}^{D} p_{k,j} x_j + r_k\\right)$$
               Koefisien $\\{p_{k,1}, \\dots, p_{k,D}, r_k\\}$ disebut **parameter konsekuen** (*linear parameters*).

            5. **Lapisan 5 — Agregasi Output (Node Tetap $\\Sigma$):**
               Menjumlahkan seluruh kontribusi aturan menjadi prediksi skor ISPU akhir:
               $$\\hat{y} = \\sum_{k=1}^{R} \\bar{w}_k f_k$$

            ---
            ### Algoritma *Hybrid Learning* (Jang, 1993)
            Pada setiap epoch pelatihan, ANFIS menggabungkan dua teknik optimasi dalam satu siklus:
            - **Forward Pass (Ridge Least Squares Estimator / LSE):** Parameter premis ditahan tetap sehingga keluaran $\\hat{y} = A\\theta$ bersifat **linear** terhadap vektor parameter konsekuen $\\theta$. Solusi global optimal diperoleh langsung dengan regularisasi Tikhonov/Ridge:
              $$\\theta^* = (A^T A + \\lambda I)^{-1} A^T y$$
            - **Backward Pass (Backpropagation / Gradient Descent):** Parameter konsekuen ditahan tetap, kemudian gradien fungsi rugi $L = \\frac{1}{2N}\\sum_{n=1}^{N}(\\hat{y}_n - y_n)^2$ dirambatkan mundur dari Lapisan 5 hingga Lapisan 1 untuk memperbarui parameter premis menggunakan pembaruan ternormalisasi:
              $$\\alpha \\leftarrow \\alpha - \\kappa \\frac{\\nabla_\\alpha L}{\\|\\nabla_\\alpha L\\|_2}$$
              dengan $\\kappa$ disesuaikan adaptif (naik 10% bila RMSE turun 4 langkah berturut-turut, turun 10% bila berosilasi).
            """
        )

    with tab_manual:
        ex = compute_worked_manual_example()
        st.markdown("### Contoh Perhitungan Numerik Sederhana (2 Input × 2 MF = 4 Aturan)")
        st.markdown(
            "Misalkan sebuah sistem ANFIS sederhana memprediksi **ISPU** dari dua polutan: "
            "**PM2.5** ($x_1$) dan **PM10** ($x_2$), masing-masing memiliki 2 MF Gaussian "
            "(*Rendah*: $c=0.20, \\sigma=0.30$ dan *Tinggi*: $c=0.80, \\sigma=0.30$)."
        )

        st.markdown(
            f"""
            **Langkah 0 — Input Mentah & Normalisasi Min-Max:**
            - $\\text{{PM2.5}} = {ex['raw_inputs']['PM2.5']:.1f}\\ \\mu\\text{{g/m}}^3$ (rentang $[5.0, 205.0]$) $\\Rightarrow x_1 = \\frac{{85 - 5}}{{205 - 5}} = \\mathbf{{{ex['x_norm']['x1']:.2f}}}$
            - $\\text{{PM10}} = {ex['raw_inputs']['PM10']:.1f}\\ \\mu\\text{{g/m}}^3$ (rentang $[20.0, 340.0]$) $\\Rightarrow x_2 = \\frac{{180 - 20}}{{340 - 20}} = \\mathbf{{{ex['x_norm']['x2']:.2f}}}$

            **Langkah 1 — Layer 1 (Fuzzifikasi Gaussian):**
            - $\\mu_{{\\text{{PM2.5, Rendah}}}}(0.40) = \\exp\\left(-\\frac{{(0.40 - 0.20)^2}}{{2 \\cdot 0.30^2}}\\right) = \\mathbf{{{ex['layer1']['mu_x1_low']:.4f}}}$
            - $\\mu_{{\\text{{PM2.5, Tinggi}}}}(0.40) = \\exp\\left(-\\frac{{(0.40 - 0.80)^2}}{{2 \\cdot 0.30^2}}\\right) = \\mathbf{{{ex['layer1']['mu_x1_high']:.4f}}}$
            - $\\mu_{{\\text{{PM10, Rendah}}}}(0.50) = \\exp\\left(-\\frac{{(0.50 - 0.20)^2}}{{2 \\cdot 0.30^2}}\\right) = \\mathbf{{{ex['layer1']['mu_x2_low']:.4f}}}$
            - $\\mu_{{\\text{{PM10, Tinggi}}}}(0.50) = \\exp\\left(-\\frac{{(0.50 - 0.80)^2}}{{2 \\cdot 0.30^2}}\\right) = \\mathbf{{{ex['layer1']['mu_x2_high']:.4f}}}$

            **Langkah 2, 3, 4 — Firing Strength ($w_k$), Normalisasi ($\\bar{{w}}_k$, dengan $\\sum w_k = {ex['w_sum']:.4f}$), dan Konsekuen ($f_k$):**
            """
        )

        st.dataframe(ex["steps_df"].round(4), width="stretch", hide_index=True)

        st.success(
            f"**Langkah 5 — Layer 5 (Agregasi Output):** "
            f"ŷ = {ex['weighted_f'][0]:.4f} + {ex['weighted_f'][1]:.4f} + {ex['weighted_f'][2]:.4f} + {ex['weighted_f'][3]:.4f} "
            f"= **{ex['y_manual']:.4f}** (Verifikasi mesin kelas ANFIS: **{ex['y_engine']:.4f}** — selisih {abs(ex['y_manual'] - ex['y_engine']):.1e})."
        )

    with tab_guide:
        st.markdown("### Library Python yang Digunakan & Fungsinya")
        lib_df = pd.DataFrame(
            [
                {"Library": "NumPy", "Fungsi Utama dalam Proyek": "Komputasi matriks, tensor Lapisan 1–5 ANFIS, aljabar linear Ridge LSE, dan kalkulus gradien Backpropagation."},
                {"Library": "Pandas", "Fungsi Utama dalam Proyek": "Pembacaan dataset CSV, pembersihan data, inspeksi statistik deskriptif, serta pembuatan tabel aturan & trace."},
                {"Library": "Scikit-learn", "Fungsi Utama dalam Proyek": "Pembagian data terstratifikasi (train_test_split), K-Means pada Scatter Partitioning, metrik evaluasi, serta model pembanding (Linear Regression, MLP, Random Forest)."},
                {"Library": "Plotly", "Fungsi Utama dalam Proyek": "Visualisasi interaktif diagram arsitektur ANFIS, kurva fungsi keanggotaan, grafik training, scatter/residual, confusion matrix, dan komparasi model."},
                {"Library": "Streamlit", "Fungsi Utama dalam Proyek": "Membangun antarmuka pengguna grafis (GUI) berbasis web multi-halaman secara interaktif dan responsif."},
            ]
        )
        st.dataframe(lib_df, width="stretch", hide_index=True)

        st.markdown("### Petunjuk Penggunaan Aplikasi")
        st.markdown(
            """
            1. **Halaman 1 (Beranda):** Tinjau ringkasan dataset aktif, status model ANFIS, dan ambang batas kategori ISPU resmi.
            2. **Halaman 2 (Input & Eksplorasi Data):** Unggah dataset CSV baru (atau gunakan dataset bawaan 1.200 baris), lihat korelasi antarpolutan, serta uji prediksi cepat melalui input manual atau tombol skenario preset.
            3. **Halaman 3 (Pelatihan ANFIS 5-Layer):** Eksperimen dengan *Grid* vs *Scatter Partitioning*, kurva *Gaussian* vs *Gbell*, jumlah epoch, dan step-size $\\kappa$, lalu amati pergeseran kurva fungsi keanggotaan sebelum dan sesudah pelatihan.
            4. **Halaman 4 (Hasil & Trace Layer 1–5):** Pilih sampel uji atau input manual terakhir untuk melihat rincian angka Lapisan 1–5 serta mengunduh hasil prediksi uji dalam format CSV.
            5. **Halaman 5 (Evaluasi & Komparasi):** Evaluasi performa numerik (RMSE, MAE, R², MAPE), akurasi kategori 5 kelas (*Confusion Matrix*), dan komparasi terhadap Regresi Linear Berganda, MLP, dan Random Forest.
            """
        )

        st.markdown("### Daftar Pustaka Utama")
        st.markdown(
            """
            - Jang, J.-S. R. (1993). *ANFIS: Adaptive-Network-Based Fuzzy Inference System*. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665–685.
            - Takagi, T., & Sugeno, M. (1985). *Fuzzy Identification of Systems and Its Applications to Modeling and Control*. IEEE Transactions on Systems, Man, and Cybernetics, SMC-15(1), 116–132.
            - Kementerian Lingkungan Hidup dan Kehutanan Republik Indonesia. (2020). *Peraturan Menteri LHK Nomor P.14/MENLHK/SETJEN/KUM.1/7/2020 tentang Indeks Standar Pencemar Udara (ISPU)*.
            """
        )
