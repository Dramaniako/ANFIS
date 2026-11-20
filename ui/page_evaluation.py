"""Halaman 5: Evaluasi Kinerja ANFIS, Diagnostik Residual, dan Komparasi Baseline."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evaluation import generate_system_analysis
from src.visualization import (
    plot_actual_vs_predicted_scatter,
    plot_actual_vs_predicted_series,
    plot_baseline_comparison,
    plot_confusion_matrix,
    plot_residuals,
)
from ui.styles import render_kpi_card


def render_page_evaluation() -> None:
    """Merender halaman evaluasi kuantitatif, confusion matrix, dan komparasi metode."""
    st.title("Evaluasi Sistem & Komparasi Baseline")
    st.caption(
        "Analisis akurasi regresi kontinu, ketepatan klasifikasi 5 kategori ISPU, "
        "diagnostik error, serta perbandingan ANFIS terhadap metode non-fuzzy."
    )

    split_data = st.session_state["split_data"]
    model = st.session_state["anfis_model"]
    train_metrics = st.session_state["train_metrics"]
    test_metrics = st.session_state["test_metrics"]
    test_cls = st.session_state["test_cls_metrics"]
    y_test = split_data["y_test"]
    y_pred_test = st.session_state["y_pred_test"]
    baseline_results = st.session_state["baseline_results"]
    train_time = float(st.session_state.get("training_time_s", 0.0))

    st.subheader("1. Metrik Evaluasi Model ANFIS")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("RMSE Uji", f"{test_metrics['RMSE']:.3f}", f"RMSE Latih: {train_metrics['RMSE']:.3f}", "#0f766e")
    with c2:
        render_kpi_card("MAE Uji", f"{test_metrics['MAE']:.3f}", f"MAPE Uji: {test_metrics['MAPE']:.2f}%", "#2563eb")
    with c3:
        render_kpi_card("Koefisien R² Uji", f"{test_metrics['R2']:.4f}", f"R² Latih: {train_metrics['R2']:.4f}", "#059669")
    with c4:
        render_kpi_card(
            "Akurasi Kategori ISPU",
            f"{test_cls['accuracy'] * 100.0:.1f}%",
            f"Macro F1: {test_cls['macro_f1'] * 100.0:.1f}%",
            "#d97706",
        )

    with st.expander("Ringkasan Metrik Lengkap (Data Latih vs Data Uji)", expanded=False):
        summary_df = pd.DataFrame(
            [
                {"Himpunan": "Data Latih (Train)", **train_metrics},
                {"Himpunan": "Data Uji (Test)", **test_metrics},
            ]
        )
        st.dataframe(summary_df.round(4), width="stretch", hide_index=True)

    st.subheader("2. Visualisasi Prediksi Aktual vs Sistem")
    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(plot_actual_vs_predicted_scatter(y_test, y_pred_test), width="stretch")
    with p2:
        st.plotly_chart(plot_actual_vs_predicted_series(y_test, y_pred_test, max_points=80), width="stretch")

    st.subheader("3. Diagnostik Residual & Confusion Matrix Kategori ISPU")
    r1, r2 = st.columns([1.25, 1.0])
    with r1:
        st.plotly_chart(plot_residuals(y_test, y_pred_test), width="stretch")
    with r2:
        st.plotly_chart(
            plot_confusion_matrix(test_cls["confusion_matrix"], test_cls["labels"]),
            width="stretch",
        )

    st.markdown("---")
    st.subheader("4. Komparasi Kinerja ANFIS vs Model Baseline")
    st.markdown(
        "Seluruh model dievaluasi pada pembagian data latih dan data uji yang **sama persis** "
        "agar perbandingan bersifat adil."
    )

    anfis_row = {
        "Model": f"ANFIS 5-Layer ({model.partition_type.title()})",
        "RMSE": test_metrics["RMSE"],
        "MAE": test_metrics["MAE"],
        "R2": test_metrics["R2"],
        "MAPE": test_metrics["MAPE"],
        "Akurasi_Kategori": test_cls["accuracy"] * 100.0,
        "Macro_F1": test_cls["macro_f1"] * 100.0,
        "Waktu_Latih_s": train_time,
    }
    comp_rows = [anfis_row] + [
        {k: v for k, v in row.items() if k != "y_pred_test"}
        for row in baseline_results
    ]
    comp_df = pd.DataFrame(comp_rows)

    st.plotly_chart(plot_baseline_comparison(comp_df), width="stretch")

    display_df = comp_df.rename(
        columns={
            "R2": "R²",
            "MAPE": "MAPE (%)",
            "Akurasi_Kategori": "Akurasi Kategori (%)",
            "Macro_F1": "Macro F1 (%)",
            "Waktu_Latih_s": "Waktu Latih (dtk)",
        }
    )
    st.dataframe(display_df.round(3), width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("5. Interpretasi Kualitas, Kelebihan, dan Keterbatasan Sistem")
    analysis = generate_system_analysis(
        anfis_test_reg=test_metrics,
        anfis_train_reg=train_metrics,
        anfis_test_cls=test_cls,
        baseline_results=baseline_results,
        model_summary=model.summary(),
    )

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.markdown("**Evaluasi Kualitas Hasil**")
        for item in analysis["kualitas"]:
            st.markdown(f"- {item}")
    with col_a2:
        st.markdown("**Kelebihan Pendekatan ANFIS**")
        for item in analysis["kelebihan"]:
            st.markdown(f"- {item}")
    with col_a3:
        st.markdown("**Keterbatasan & Catatan Pengujian**")
        for item in analysis["keterbatasan"]:
            st.markdown(f"- {item}")
