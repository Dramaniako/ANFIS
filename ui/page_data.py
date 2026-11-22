"""Halaman 2: Input Data (Dataset CSV + Input Manual) dan Eksplorasi Polutan."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ispu_standard import (
    ISPU_BREAKPOINTS,
    POLLUTANT_ORDER,
    POLLUTANT_SPECS,
    PRESET_SCENARIOS,
    PRESET_SCENARIOS_SUBINDEX,
)
from src.preprocessing import auto_map_columns, validate_and_clean_dataframe
from src.visualization import plot_data_exploration_categories, plot_data_exploration_corr
from ui.state import apply_bundle_to_session, execute_training_pipeline, run_manual_prediction
from ui.styles import render_ispu_badge


def _apply_preset_callback(preset_name: str, feature_cols: list[str], presets: dict) -> None:
    preset_vals = presets[preset_name]["values"]
    for col in feature_cols:
        key_lower = col.lower()
        if key_lower in preset_vals:
            val = float(preset_vals[key_lower])
            st.session_state["manual_inputs"][col] = val
            st.session_state[f"manual_field_{col}"] = val


def render_page_data() -> None:
    """Merender halaman pengelolaan dataset CSV, eksplorasi statistik, dan input manual."""
    st.title("Input & Eksplorasi Data Polusi Udara")
    st.caption("Kelola dataset CSV untuk pelatihan atau uji konsentrasi polutan secara manual.")

    tab_dataset, tab_manual = st.tabs(["Dataset & Unggah CSV", "Input Manual & Prediksi Cepat"])

    with tab_dataset:
        _render_dataset_tab()

    with tab_manual:
        _render_manual_tab()


def _render_dataset_tab() -> None:
    df_clean: pd.DataFrame = st.session_state["df_clean"]
    split_data = st.session_state["split_data"]
    clean_report = st.session_state["clean_report"]

    st.subheader("1. Unggah Dataset CSV Kustom (Opsional)")
    with st.expander("Unggah CSV baru atau ganti pemetaan kolom dataset", expanded=False):
        uploaded_file = st.file_uploader(
            "Pilih file CSV kualitas udara (harus memiliki kolom polutan dan kolom target skor ISPU/AQI)",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success(f"File '{uploaded_file.name}' berhasil dibaca ({len(uploaded_df)} baris, {len(uploaded_df.columns)} kolom).")

                all_cols = list(uploaded_df.columns)
                auto_map = auto_map_columns(all_cols)
                default_feats = [auto_map[p] for p in POLLUTANT_ORDER if auto_map.get(p) in all_cols]
                if len(default_feats) < 2:
                    numeric_cols = list(uploaded_df.select_dtypes(include="number").columns)
                    default_feats = numeric_cols[:-1] if len(numeric_cols) >= 3 else all_cols[:2]

                default_target = auto_map.get("ispu")
                if default_target not in all_cols:
                    default_target = all_cols[-1]

                c_map1, c_map2 = st.columns(2)
                with c_map1:
                    selected_feats = st.multiselect(
                        "Pilih Kolom Fitur Input (minimal 2)",
                        options=all_cols,
                        default=default_feats,
                        key="upload_feat_cols",
                    )
                with c_map2:
                    selected_target = st.selectbox(
                        "Pilih Kolom Target Numerik (ISPU / AQI)",
                        options=all_cols,
                        index=all_cols.index(default_target),
                        key="upload_target_col",
                    )

                c_opt1, c_opt2 = st.columns(2)
                with c_opt1:
                    miss_strat = st.selectbox(
                        "Penanganan Nilai Kosong (Missing Value)",
                        options=["median", "drop"],
                        format_func=lambda v: "Imputasi Median" if v == "median" else "Hapus Baris Kosong",
                    )
                with c_opt2:
                    clip_neg = st.checkbox(
                        "Kliping nilai konsentrasi negatif menjadi 0",
                        value=True,
                    )

                if st.button("Validasi, Terapkan Dataset & Latih Ulang ANFIS", type="primary", width="stretch"):
                    try:
                        cleaned, rep = validate_and_clean_dataframe(
                            uploaded_df,
                            feature_cols=selected_feats,
                            target_col=selected_target,
                            missing_strategy=miss_strat,
                            clip_negative=clip_neg,
                        )
                        part_mode = "grid" if 2 ** len(selected_feats) <= 64 else "scatter"
                        trained = execute_training_pipeline(
                            df_clean=cleaned,
                            feature_cols=selected_feats,
                            target_col=selected_target,
                            partition_type=part_mode,
                            n_mfs=2,
                            n_rules=12,
                            epochs=40,
                        )
                        apply_bundle_to_session(
                            {
                                "df_raw": uploaded_df,
                                "df_clean": cleaned,
                                "clean_report": rep,
                                "data_source_name": f"CSV Unggahan ({uploaded_file.name})",
                                **trained,
                            }
                        )
                        st.session_state["last_manual_result"] = None
                        st.success(
                            f"Dataset '{uploaded_file.name}' divalidasi ({len(cleaned)} baris bersih) dan model ANFIS selesai dilatih!"
                        )
                        st.rerun()
                    except ValueError as exc:
                        st.error(f"Validasi dataset gagal: {exc}")
            except Exception as exc:
                st.error(f"Gagal membaca file CSV: {exc}")

    st.subheader("2. Ringkasan Dataset Aktif")
    if st.session_state.get("input_mode") == "sub_index":
        st.info(
            "Dataset aktif terdeteksi menggunakan skala **Sub-Indeks ISPU** per polutan "
            "(seperti data SPKU Satu Data DKI Jakarta di mana `max` merupakan nilai sub-indeks tertinggi). "
            "Baris historis pra-2021 tanpa pengukuran `PM2.5` dibersihkan agar seluruh data latih lengkap dan autentik."
        )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sumber Data", st.session_state.get("data_source_name", "Bawaan").split("(")[0].strip())
    m2.metric("Baris Bersih", f"{clean_report['clean_rows']:,}", f"Dibuang: {clean_report['dropped_rows']}")
    m3.metric("Sel Diimputasi", f"{clean_report['imputed_cells']}", f"Negatif diperbaiki: {clean_report['negative_fixed']}")
    m4.metric("Outlier (IQR)", f"{clean_report['outlier_cells_iqr']}", "Dipertahankan (kasus ekstrem)")

    st.dataframe(df_clean.head(50), width="stretch", hide_index=True)

    with st.expander("Statistik Deskriptif & Definisi Variabel Polutan (Permen LHK 14/2020)"):
        active_cols = [*split_data["feature_cols"], split_data["target_col"]]
        st.markdown("**Statistik Deskriptif Kolom Aktif:**")
        st.dataframe(df_clean[active_cols].describe().T.round(2), width="stretch")

        st.markdown("**Penjelasan Atribut & Ambang Batas Konsentrasi 24 Jam (Permen LHK No. 14/2020):**")
        spec_rows = []
        for pkey in POLLUTANT_ORDER:
            sp = POLLUTANT_SPECS[pkey]
            bps = sp["breakpoints"]
            spec_rows.append(
                {
                    "Variabel": sp["name"],
                    "Nama Lengkap": sp["full_name"],
                    "Satuan": sp["unit"],
                    "Baik (≤50)": bps[1],
                    "Sedang (≤100)": bps[2],
                    "Tidak Sehat (≤200)": bps[3],
                    "Sangat Tidak Sehat (≤300)": bps[4],
                    "Berbahaya (>300)": f"> {bps[4]} (ISPU 500 = {bps[5]})",
                    "Dampak Kesehatan": sp["health_impact"],
                }
            )
        st.dataframe(pd.DataFrame(spec_rows), width="stretch", hide_index=True)
        st.caption(f"Breakpoint Nilai Indeks ISPU resmi: {', '.join(str(int(x)) for x in ISPU_BREAKPOINTS)}.")

    st.subheader("3. Visualisasi Eksplorasi Data")
    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(
            plot_data_exploration_categories(df_clean, target_col=split_data["target_col"]),
            width="stretch",
        )
    with v2:
        corr_cols = [*split_data["feature_cols"], split_data["target_col"]]
        st.plotly_chart(plot_data_exploration_corr(df_clean, corr_cols), width="stretch")


def _render_manual_tab() -> None:
    split_data = st.session_state["split_data"]
    input_mode: str = st.session_state.get("input_mode", "concentration")
    is_subindex = input_mode == "sub_index"
    feature_cols: list[str] = split_data["feature_cols"]
    feature_bounds: dict[str, tuple[float, float]] = split_data["feature_bounds"]
    manual_defaults: dict[str, float] = st.session_state.get("manual_inputs", {})

    active_presets = PRESET_SCENARIOS_SUBINDEX if is_subindex else PRESET_SCENARIOS
    st.subheader("Uji Prediksi Kualitas Udara Secara Manual")
    if is_subindex:
        st.markdown(
            "Masukkan nilai **Sub-Indeks ISPU** tiap polutan sesuai skala dataset aktif, atau klik salah satu "
            "**Skenario Preset** untuk mengisi angka contoh secara otomatis."
        )
    else:
        st.markdown(
            "Masukkan konsentrasi tiap polutan udara atau klik salah satu **Skenario Preset** "
            "untuk mengisi angka contoh secara otomatis."
        )

    # Tombol Skenario Preset
    preset_cols = st.columns(len(active_presets))
    for col_ui, (pname, pinfo) in zip(preset_cols, active_presets.items()):
        with col_ui:
            st.button(
                pname,
                help=pinfo["description"],
                on_click=_apply_preset_callback,
                args=(pname, feature_cols, active_presets),
                width="stretch",
            )

    st.markdown("#### Masukkan Nilai Sub-Indeks Polutan" if is_subindex else "#### Masukkan Nilai Konsentrasi Polutan")
    input_grid_cols = st.columns(3)
    collected_inputs: dict[str, float] = {}

    for idx, col in enumerate(feature_cols):
        spec = POLLUTANT_SPECS.get(col.lower())
        low_b, high_b = feature_bounds[col]
        fallback_val = (
            round((low_b + high_b) / 2.0, 1)
            if (is_subindex or spec is None)
            else float(spec["default_value"])
        )
        default_val = float(manual_defaults.get(col, fallback_val))
        field_key = f"manual_field_{col}"
        if field_key not in st.session_state:
            st.session_state[field_key] = default_val

        unit_label = "Sub-Indeks ISPU" if is_subindex else (spec["unit"] if spec else "")
        name_label = spec["name"] if spec else col
        label = f"{name_label} ({unit_label})" if unit_label else name_label
        if spec:
            help_txt = f"{spec['description']} Rentang data latih: [{low_b:.1f}, {high_b:.1f}] {unit_label}."
        else:
            help_txt = f"Rentang data latih: [{low_b:.2f}, {high_b:.2f}]"

        with input_grid_cols[idx % 3]:
            # Tidak mengunci min_value=0 agar pesan validasi kesalahan input negatif dapat didemonstrasikan
            val = st.number_input(
                label,
                step=1.0,
                format="%.2f",
                help=help_txt,
                key=field_key,
            )
            collected_inputs[col] = val

    if st.button("Hitung Prediksi ISPU (ANFIS)", type="primary", width="stretch"):
        res = run_manual_prediction(collected_inputs)
        if not res["ok"]:
            for err in res["errors"]:
                st.error(err)
        else:
            st.session_state["last_manual_result"] = res

    res = st.session_state.get("last_manual_result")
    if res is not None and res.get("ok"):
        for warn in res.get("warnings", []):
            st.warning(warn)

        st.markdown("---")
        st.subheader("Hasil Prediksi Manual")
        col_res1, col_res2 = st.columns([1.35, 1.0])
        with col_res1:
            render_ispu_badge(res["pred_score"], title="Prediksi Model ANFIS 5-Layer")
        with col_res2:
            if res.get("official") is not None:
                off = res["official"]
                metric_title = (
                    "ISPU Analitis (Maksimum Sub-Indeks)"
                    if is_subindex
                    else "ISPU Analitis (Rumus Permen LHK 14/2020)"
                )
                st.metric(
                    metric_title,
                    f"{off['ispu']:.2f}",
                    f"Kategori: {off['kategori']} • Polutan Kritis: {off['polutan_kritis']}",
                )
                diff = res["pred_score"] - off["ispu"]
                st.caption(
                    f"Selisih ANFIS terhadap rumus analitis murni: **{diff:+.2f} poin** "
                    "(ANFIS juga memodelkan pola interaksi antarpolutan dari data historis)."
                )
            st.info(
                "Untuk menelusuri perhitungan rinci **Layer 1 s.d. Layer 5** pada input ini, "
                "buka halaman **4. Hasil & Trace Layer 1–5** di sidebar."
            )
