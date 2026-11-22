"""Halaman 4: Hasil Prediksi, Trace Perhitungan Layer 1–5, dan Tabel Data Uji."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.ispu_standard import (
    POLLUTANT_SPECS,
    classify_ispu,
    compute_ispu_from_subindices,
    compute_ispu_official,
)
from src.visualization import plot_rule_contributions
from ui.styles import render_ispu_badge


def render_page_results() -> None:
    """Merender halaman transparansi perhitungan Layer 1–5 dan tabel hasil prediksi uji."""
    st.title("Hasil Prediksi & Trace Perhitungan Layer 1–5")
    st.caption(
        "Telusuri setiap angka antara dari input mentah hingga agregasi output akhir "
        "pada arsitektur 5-Lapisan ANFIS."
    )

    split_data = st.session_state["split_data"]
    input_mode: str = st.session_state.get("input_mode", "concentration")
    is_subindex = input_mode == "sub_index"
    model = st.session_state["anfis_model"]
    scaler = split_data["scaler"]
    feature_cols: list[str] = split_data["feature_cols"]
    X_test_raw: np.ndarray = split_data["X_test_raw"]
    X_test: np.ndarray = split_data["X_test"]
    y_test: np.ndarray = split_data["y_test"]
    y_pred_test: np.ndarray = st.session_state["y_pred_test"]

    def _compute_ref_ispu(vals: dict[str, float]) -> dict | None:
        if not vals:
            return None
        return compute_ispu_from_subindices(vals) if is_subindex else compute_ispu_official(vals)

    st.subheader("1. Pilih Sampel untuk Ditelusuri")
    source_mode = st.radio(
        "Sumber Sampel:",
        options=["Sampel Data Uji", "Input Manual Terakhir"],
        horizontal=True,
    )

    actual_score: float | None = None
    official_info: dict | None = None

    if source_mode == "Sampel Data Uji":
        sample_idx = st.slider(
            "Pilih Indeks Sampel Data Uji",
            min_value=0,
            max_value=len(y_test) - 1,
            value=0,
            step=1,
        )
        x_raw = X_test_raw[sample_idx]
        x_scaled = X_test[sample_idx]
        actual_score = float(y_test[sample_idx])

        official_vals = {
            col.lower(): float(x_raw[j])
            for j, col in enumerate(feature_cols)
            if col.lower() in POLLUTANT_SPECS
        }
        official_info = _compute_ref_ispu(official_vals)
    else:
        manual_res = st.session_state.get("last_manual_result")
        if manual_res is not None and manual_res.get("ok"):
            x_raw = np.asarray(manual_res["x_raw"], dtype=np.float64)
            x_scaled = np.asarray(manual_res["x_scaled"], dtype=np.float64)
            official_info = manual_res.get("official")
        else:
            manual_dict = st.session_state.get("manual_inputs", {})
            x_raw = np.array(
                [
                    float(
                        manual_dict.get(
                            col,
                            round(float(np.median(X_test_raw[:, j])), 1)
                            if (is_subindex or col.lower() not in POLLUTANT_SPECS)
                            else POLLUTANT_SPECS[col.lower()]["default_value"],
                        )
                    )
                    for j, col in enumerate(feature_cols)
                ],
                dtype=np.float64,
            )
            x_scaled = scaler.transform(x_raw)
            official_vals = {
                col.lower(): float(x_raw[j])
                for j, col in enumerate(feature_cols)
                if col.lower() in POLLUTANT_SPECS
            }
            official_info = _compute_ref_ispu(official_vals)
            st.info("Menampilkan trace untuk nilai input manual default (ubah nilai pada halaman 2. Input & Eksplorasi Data).")

    trace = model.forward_trace(x_scaled_1d=x_scaled, x_raw_1d=x_raw)
    pred_score = trace["y_pred"]

    # Ringkasan hasil sampel terpilih
    c_badge, c_ref = st.columns([1.4, 1.0])
    with c_badge:
        render_ispu_badge(pred_score, title="Output Akhir Layer 5 (Prediksi ANFIS)")
    with c_ref:
        if actual_score is not None:
            abs_err = abs(actual_score - pred_score)
            actual_cat = classify_ispu(actual_score)["name"]
            st.metric("ISPU Aktual (Ground Truth)", f"{actual_score:.2f}", f"Kategori Aktual: {actual_cat}")
            st.metric("Error Absolut |Aktual - Prediksi|", f"{abs_err:.2f} poin ISPU")
        elif official_info is not None:
            st.metric(
                "ISPU Analitis (Permen LHK 14/2020)",
                f"{official_info['ispu']:.2f}",
                f"Polutan Kritis: {official_info['polutan_kritis']} ({official_info['kategori']})",
            )

    st.markdown("---")
    st.subheader("2. Trace Komputasi ANFIS Langkah demi Langkah")

    # Tahap 0: Normalisasi
    with st.expander("Tahap 0 — Input Mentah & Normalisasi Min-Max [0, 1]", expanded=True):
        norm_rows = []
        for j, col in enumerate(feature_cols):
            spec = POLLUTANT_SPECS.get(col.lower())
            unit_str = "Sub-Indeks ISPU" if is_subindex else (spec["unit"] if spec else "-")
            norm_rows.append(
                {
                    "Variabel": spec["name"] if spec else col,
                    "Satuan": unit_str,
                    "Nilai Mentah (x)": float(x_raw[j]),
                    "Min Latih": float(scaler.data_min_[j]),
                    "Max Latih": float(scaler.data_max_[j]),
                    "Ternormalisasi (x_norm)": float(x_scaled[j]),
                }
            )
        st.dataframe(pd.DataFrame(norm_rows).round(4), width="stretch", hide_index=True)

    # Layer 1: Fuzzifikasi
    with st.expander("Layer 1 — Fuzzifikasi (Derajat Keanggotaan μ)", expanded=True):
        st.markdown(
            "Menghitung derajat keanggotaan setiap variabel ternormalisasi terhadap fungsi keanggotaan "
            f"**{model.mf_type.title()}** terlatih."
        )
        st.dataframe(trace["mf_table"].round(4), width="stretch", hide_index=True)

    # Layer 2, 3, 4: Aturan, Normalisasi, Konsekuen
    with st.expander("Layer 2, 3 & 4 — Firing Strength (w), Normalisasi (w̄), dan Output Konsekuen (f_k)", expanded=True):
        rule_df = trace["rule_table"]
        st.markdown(
            f"Total firing strength seluruh aturan: $\\sum_{{k=1}}^{{{model.n_rules}}} w_k = {trace['w_sum']:.6f}$. "
            "Tabel berikut diurutkan dari aturan dengan bobot aktif $\\bar{w}_k$ terbesar:"
        )
        st.plotly_chart(plot_rule_contributions(rule_df, top_k=min(10, len(rule_df))), width="stretch")
        st.dataframe(rule_df.round(5), width="stretch", hide_index=True)

    # Layer 5: Agregasi Output
    with st.expander("Layer 5 — Agregasi Output Akhir (Σ w̄_k · f_k)", expanded=True):
        sum_wbar = float(rule_df["w_bar (Layer 3)"].sum())
        sum_contrib = float(rule_df["w_bar * f_k"].sum())
        st.markdown(
            f"""
            - Penjumlahan seluruh bobot ternormalisasi: $\\sum \\bar{{w}}_k = {sum_wbar:.6f}$
            - Penjumlahan seluruh kontribusi aturan: $\\hat{{y}} = \\sum_{{k=1}}^{{{model.n_rules}}} \\bar{{w}}_k f_k = \\mathbf{{{sum_contrib:.4f}}}$
            - Kategori ISPU hasil prediksi: **{classify_ispu(sum_contrib)['name']}**
            """
        )

    st.markdown("---")
    st.subheader("3. Tabel Hasil Prediksi Seluruh Data Uji")

    test_table_data: dict[str, object] = {}
    for j, col in enumerate(feature_cols):
        test_table_data[col] = np.round(X_test_raw[:, j], 2)
    test_table_data["ISPU_Aktual"] = np.round(y_test, 2)
    test_table_data["ISPU_Prediksi_ANFIS"] = np.round(y_pred_test, 2)
    test_table_data["Error_Absolut"] = np.round(np.abs(y_test - y_pred_test), 2)
    test_table_data["Kategori_Aktual"] = [classify_ispu(v)["name"] for v in y_test]
    test_table_data["Kategori_Prediksi"] = [classify_ispu(v)["name"] for v in y_pred_test]

    df_test_results = pd.DataFrame(test_table_data)
    st.dataframe(df_test_results, width="stretch", hide_index=True)

    csv_bytes = df_test_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Unduh Tabel Hasil Prediksi Data Uji (CSV)",
        data=csv_bytes,
        file_name="prediksi_ispu_anfis.csv",
        mime="text/csv",
    )
