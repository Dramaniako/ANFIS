"""Halaman 3: Konfigurasi Arsitektur dan Pelatihan ANFIS 5-Layer (Hybrid Learning)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.anfis import MAX_GRID_RULES
from src.visualization import plot_membership_functions, plot_training_curves
from ui.state import apply_bundle_to_session, execute_training_pipeline
from ui.styles import render_kpi_card


def render_page_training() -> None:
    """Merender halaman konfigurasi parameter, pelatihan Hybrid Learning, dan inspeksi MF."""
    st.title("Pelatihan Model ANFIS 5-Layer")
    st.caption(
        "Atur strategi partisi aturan fuzzy, jenis fungsi keanggotaan, dan parameter Hybrid Learning "
        "(Forward Ridge LSE + Backward Gradient Descent)."
    )

    df_clean: pd.DataFrame = st.session_state["df_clean"]
    split_data = st.session_state["split_data"]
    target_col: str = split_data["target_col"]

    numeric_candidates = [
        col
        for col in df_clean.select_dtypes(include="number").columns
        if col != target_col
    ]
    if not numeric_candidates:
        numeric_candidates = list(split_data["feature_cols"])

    with st.container(border=True):
        st.subheader("1. Konfigurasi Arsitektur & Hiperparameter")
        c1, c2, c3 = st.columns(3)

        with c1:
            selected_features = st.multiselect(
                "Fitur Input Polutan (min. 2)",
                options=numeric_candidates,
                default=[c for c in split_data["feature_cols"] if c in numeric_candidates],
            )
            partition_type = st.selectbox(
                "Metode Partisi Ruang Input",
                options=["grid", "scatter"],
                format_func=lambda v: (
                    "Grid Partitioning (Klasik Jang)"
                    if v == "grid"
                    else "Scatter Partitioning (K-Means Clustering)"
                ),
                help="Grid membentuk aturan M^D penuh; Scatter memakai K-Means untuk menjaga jumlah aturan ringkas.",
            )
            mf_type = st.selectbox(
                "Jenis Fungsi Keanggotaan (Layer 1)",
                options=["gaussian", "gbell"],
                format_func=lambda v: "Gaussian (c, σ)" if v == "gaussian" else "Generalized Bell (a, b, c)",
            )

        with c2:
            if partition_type == "grid":
                n_mfs = st.selectbox(
                    "Jumlah MF per Input (M)",
                    options=[2, 3],
                    index=0,
                    help="Dengan D input dan M fungsi keanggotaan, jumlah aturan fuzzy adalah R = M^D.",
                )
                n_rules = 12
                est_rules = n_mfs ** max(len(selected_features), 1)
                if est_rules > MAX_GRID_RULES:
                    st.warning(
                        f"Kombinasi {len(selected_features)} fitur × {n_mfs} MF = {est_rules} aturan "
                        f"(melebihi batas {MAX_GRID_RULES}). Kurangi fitur atau pilih Scatter Partitioning."
                    )
                else:
                    st.info(f"Jumlah aturan terbentuk: **{est_rules} aturan fuzzy** ({n_mfs}^{len(selected_features)}).")
            else:
                n_mfs = 2
                n_rules = st.slider(
                    "Jumlah Aturan / Cluster K-Means (R)",
                    min_value=4,
                    max_value=36,
                    value=12,
                    step=2,
                )
                st.info(f"Scatter Partitioning akan membentuk **{n_rules} aturan fuzzy** berbasis pusat cluster.")

            test_pct = st.slider("Persentase Data Uji (%)", min_value=10, max_value=35, value=20, step=5)

        with c3:
            epochs = st.slider("Jumlah Epoch Hybrid Learning", min_value=10, max_value=150, value=40, step=5)
            learning_rate = st.select_slider(
                "Step-Size Awal Backprop (κ)",
                options=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
                value=0.02,
            )
            l2_reg = st.select_slider(
                "Regularisasi Ridge LSE (λ)",
                options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                value=1e-3,
                format_func=lambda v: f"{v:.0e}",
            )
            random_seed = int(st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1))

        can_train = len(selected_features) >= 2 and not (
            partition_type == "grid" and (n_mfs ** len(selected_features)) > MAX_GRID_RULES
        )

        if st.button(
            "Jalankan Pelatihan ANFIS (Hybrid Learning)",
            type="primary",
            disabled=not can_train,
            width="stretch",
        ):
            progress_bar = st.progress(0, text="Memulai pelatihan ANFIS...")
            status_text = st.empty()

            def _on_epoch(ep: int, total: int, info: dict[str, float]) -> None:
                pct = int(round((ep / total) * 100))
                progress_bar.progress(pct, text=f"Epoch {ep}/{total} — RMSE Latih: {info['train_rmse']:.3f}")
                status_text.caption(
                    f"Epoch {ep}/{total} | RMSE Latih: {info['train_rmse']:.3f} | "
                    f"RMSE Uji: {info['val_rmse']:.3f} | κ: {info['kappa']:.5f}"
                )

            try:
                trained = execute_training_pipeline(
                    df_clean=df_clean,
                    feature_cols=selected_features,
                    target_col=target_col,
                    test_size=test_pct / 100.0,
                    partition_type=partition_type,
                    n_mfs=n_mfs,
                    n_rules=n_rules,
                    mf_type=mf_type,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    l2_reg=l2_reg,
                    random_state=random_seed,
                    progress_callback=_on_epoch,
                )
                apply_bundle_to_session(trained)
                st.session_state["last_manual_result"] = None
                st.success(
                    f"Pelatihan ANFIS selesai dalam {trained['training_time_s']:.2f} detik! "
                    f"RMSE Uji = {trained['test_metrics']['RMSE']:.2f}, R² Uji = {trained['test_metrics']['R2']:.4f}."
                )
            except Exception as exc:
                st.error(f"Terjadi kesalahan saat pelatihan: {exc}")

    # Tampilkan Hasil Model Aktif
    model = st.session_state["anfis_model"]
    active_split = st.session_state["split_data"]
    summary = model.summary()
    test_metrics = st.session_state["test_metrics"]
    train_metrics = st.session_state["train_metrics"]
    train_time = st.session_state.get("training_time_s", 0.0)

    st.markdown("---")
    st.subheader("2. Ringkasan Model ANFIS Terlatih")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            "Jumlah Aturan Fuzzy",
            f"{summary['n_rules']} Aturan",
            f"Mode: {summary['partition_type'].title()} ({summary['mf_type']})",
            accent="#0f766e",
        )
    with k2:
        render_kpi_card(
            "Total Parameter Adaptif",
            f"{summary['total_params']:,}",
            f"Premis: {summary['n_premise_params']} | Konsekuen: {summary['n_consequent_params']}",
            accent="#2563eb",
        )
    with k3:
        render_kpi_card(
            "RMSE Latih / Uji",
            f"{train_metrics['RMSE']:.2f} / {test_metrics['RMSE']:.2f}",
            f"R² Uji: {test_metrics['R2']:.4f}",
            accent="#059669",
        )
    with k4:
        render_kpi_card(
            "Waktu Pelatihan",
            f"{train_time:.2f} dtk",
            f"{summary['epochs_run']} Epoch selesai",
            accent="#d97706",
        )

    st.subheader("3. Kurva Konvergensi Hybrid Learning")
    history_df = pd.DataFrame(model.history_)
    st.plotly_chart(plot_training_curves(history_df), width="stretch")

    st.subheader("4. Visualisasi Fungsi Keanggotaan (Sebelum vs Sesudah Training)")
    mf_col1, mf_col2 = st.columns([1.2, 1.0])
    with mf_col1:
        chosen_feat_idx = st.selectbox(
            "Pilih Variabel Input untuk Inspeksi Kurva MF",
            options=list(range(model.n_inputs)),
            format_func=lambda idx: model.feature_names[idx],
        )
    with mf_col2:
        show_init = st.checkbox("Tampilkan kurva awal sebelum optimasi (garis putus-putus)", value=True)

    st.plotly_chart(
        plot_membership_functions(
            model,
            feature_idx=chosen_feat_idx,
            scaler=active_split["scaler"],
            show_initial=show_init,
        ),
        width="stretch",
    )

    st.subheader("5. Inspeksi Parameter Premis & Basis Aturan Fuzzy")
    tab_premise, tab_rules = st.tabs(
        ["Parameter Premis (Awal vs Terlatih)", "Basis Aturan Fuzzy & Konsekuen Takagi-Sugeno"]
    )
    with tab_premise:
        st.dataframe(model.get_premise_params_dataframe().round(4), width="stretch", hide_index=True)
    with tab_rules:
        st.dataframe(model.get_rules_dataframe().round(4), width="stretch", hide_index=True)
