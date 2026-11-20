"""Halaman 1: Beranda dan ringkasan sistem AirFuzzy-ANFIS."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ispu_standard import ISPU_CATEGORIES
from ui.styles import render_kpi_card


def render_page_home() -> None:
    """Merender halaman utama aplikasi beserta KPI dan alur kerja sistem."""
    st.markdown(
        """
        <div class="hero-banner">
            <h1>AirFuzzy-ANFIS: Prediksi Kualitas Udara Indonesia (ISPU)</h1>
            <p>
                Sistem cerdas berbasis kombinasi <b>Logika Fuzzy Takagi-Sugeno</b> dan
                <b>Jaringan Saraf Tiruan Adaptif 5-Lapisan (ANFIS)</b> dengan algoritma
                <i>Hybrid Learning</i> (Least Squares Estimator + Backpropagation) untuk
                memprediksi dan menginterpretasikan Indeks Standar Pencemar Udara (ISPU)
                berdasarkan <b>Permen LHK No. P.14/MENLHK/SETJEN/KUM.1/7/2020</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_clean = st.session_state["df_clean"]
    split_data = st.session_state["split_data"]
    model = st.session_state["anfis_model"]
    test_metrics = st.session_state["test_metrics"]
    test_cls = st.session_state["test_cls_metrics"]
    summary = model.summary()

    # Baris KPI Utama
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Dataset Aktif",
            f"{len(df_clean):,} Baris",
            st.session_state.get("data_source_name", "Dataset ISPU"),
            accent="#0f766e",
        )
    with c2:
        render_kpi_card(
            "Arsitektur ANFIS",
            f"{summary['n_rules']} Aturan",
            f"{summary['n_inputs']} Input • {summary['partition_type'].title()} ({summary['mf_type']})",
            accent="#2563eb",
        )
    with c3:
        render_kpi_card(
            "Kinerja Uji (R² / RMSE)",
            f"{test_metrics['R2']:.4f}",
            f"RMSE Uji: {test_metrics['RMSE']:.2f} poin ISPU",
            accent="#059669",
        )
    with c4:
        render_kpi_card(
            "Ketepatan Kategori",
            f"{test_cls['accuracy'] * 100.0:.1f}%",
            f"Dari {len(split_data['y_test'])} sampel uji (5 kategori)",
            accent="#d97706",
        )

    st.markdown("---")

    col_left, col_right = st.columns([1.25, 1.0], gap="large")
    with col_left:
        st.subheader("Latar Belakang & Mengapa ANFIS 5-Layer?")
        st.markdown(
            """
            Polusi udara perkotaan dan kawasan industri merupakan tantangan kesehatan publik yang kompleks.
            Penilaian **ISPU** melibatkan enam polutan utama (**PM10, PM2.5, SO₂, CO, O₃, NO₂**)
            dengan transisi ambang batas konsentrasi yang bersifat kontinu dan saling berinteraksi.

            **Integrasi Fuzzy + Neural Network pada ANFIS mengatasi keterbatasan masing-masing metode tunggal:**
            1. **Logika Fuzzy Takagi-Sugeno** menyediakan penalaran berbasis aturan *IF-THEN* dan fungsi
               keanggotaan linguistik (*Rendah, Sedang, Tinggi*) yang mudah dipahami oleh manusia.
            2. **Jaringan Saraf Tiruan (Backpropagation + Ridge LSE)** mengoptimasi pusat/lebar kurva keanggotaan
               (parameter premis) dan koefisien fungsi linear konsekuen secara otomatis dari data historis.
            """
        )

    with col_right:
        st.subheader("Standar Kategori ISPU (Permen LHK 14/2020)")
        cat_rows = [
            {
                "Rentang ISPU": c["range_label"],
                "Kategori": c["name"],
                "Warna Resmi": c["badge_label"],
            }
            for c in ISPU_CATEGORIES
        ]
        st.dataframe(pd.DataFrame(cat_rows), width="stretch", hide_index=True)

    st.markdown("### Alur Kerja & Navigasi Aplikasi")
    steps = [
        ("Tahap 1", "Input & Eksplorasi Data", "Gunakan dataset ISPU bawaan, unggah CSV baru, atau uji masukan konsentrasi polutan manual."),
        ("Tahap 2", "Pelatihan ANFIS 5-Layer", "Atur Grid/Scatter Partitioning, jenis kurva MF, epoch, dan amati konvergensi Hybrid Learning."),
        ("Tahap 3", "Hasil & Trace Layer 1–5", "Lihat transparansi perhitungan Lapisan 1–5 pada tiap prediksi serta aturan fuzzy dominan."),
        ("Tahap 4", "Evaluasi & Komparasi", "Uji akurasi (RMSE, MAE, R², Confusion Matrix) dan bandingkan ANFIS melawan model baseline."),
    ]
    cols = st.columns(4)
    for col, (tag, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{tag}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info(
        "Gunakan menu navigasi di **Sidebar kiri** untuk berpindah antar halaman secara interaktif. "
        "Model awal sudah terlatih otomatis sehingga Anda dapat langsung mencoba prediksi maupun melatih ulang parameter."
    )
