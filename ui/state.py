"""Manajemen state sesi dan orkestrasi pipeline pelatihan/prediksi ANFIS."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import streamlit as st

from src.anfis import ANFIS
from src.data_generator import generate_ispu_dataset
from src.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    train_baseline_models,
)
from src.ispu_standard import (
    POLLUTANT_ORDER,
    POLLUTANT_SPECS,
    compute_ispu_from_subindices,
    compute_ispu_official,
    validate_manual_input,
)
from src.preprocessing import (
    auto_map_columns,
    detect_input_mode,
    prepare_train_test_data,
    validate_and_clean_dataframe,
)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "ispu_dki_all.csv"


def load_default_dataframe() -> pd.DataFrame:
    """Memuat dataset bawaan ISPU Indonesia (atau membuatnya ulang jika belum ada)."""
    if DEFAULT_DATASET_PATH.exists():
        return pd.read_csv(DEFAULT_DATASET_PATH)
    DEFAULT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_ispu_dataset(n_samples=1200, seed=42)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)
    return df


def execute_training_pipeline(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "ispu",
    test_size: float = 0.2,
    partition_type: Literal["grid", "scatter"] = "grid",
    n_mfs: int = 2,
    n_rules: int = 12,
    mf_type: Literal["gaussian", "gbell"] = "gaussian",
    epochs: int = 40,
    learning_rate: float = 0.02,
    l2_reg: float = 1e-3,
    random_state: int = 42,
    progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Menjalankan pembagian data, normalisasi, pelatihan ANFIS, dan evaluasi baseline."""
    split_data = prepare_train_test_data(
        df_clean,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    model = ANFIS(
        n_inputs=len(feature_cols),
        partition_type=partition_type,
        n_mfs=n_mfs,
        n_rules=n_rules,
        mf_type=mf_type,
        learning_rate=learning_rate,
        l2_reg=l2_reg,
        epochs=epochs,
        random_state=random_state,
        feature_names=feature_cols,
    )

    t0 = time.perf_counter()
    model.fit(
        split_data["X_train"],
        split_data["y_train"],
        X_val=split_data["X_test"],
        y_val=split_data["y_test"],
        progress_callback=progress_callback,
    )
    training_time_s = time.perf_counter() - t0

    y_pred_train = model.predict(split_data["X_train"])
    y_pred_test = model.predict(split_data["X_test"])

    train_metrics = compute_regression_metrics(split_data["y_train"], y_pred_train)
    test_metrics = compute_regression_metrics(split_data["y_test"], y_pred_test)
    test_cls_metrics = compute_classification_metrics(split_data["y_test"], y_pred_test)

    baseline_results = train_baseline_models(
        split_data["X_train"],
        split_data["y_train"],
        split_data["X_test"],
        split_data["y_test"],
        random_state=random_state,
    )
    input_mode = detect_input_mode(df_clean, feature_cols, target_col)

    return {
        "split_data": split_data,
        "anfis_model": model,
        "training_time_s": training_time_s,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_cls_metrics": test_cls_metrics,
        "baseline_results": baseline_results,
        "input_mode": input_mode,
    }


@st.cache_resource(show_spinner="Menyiapkan model awal ANFIS pada dataset ISPU Indonesia...")
def build_initial_bundle() -> dict[str, Any]:
    """Membangun state awal ter-cache agar aplikasi siap langsung digunakan saat dibuka."""
    df_raw = load_default_dataframe()
    col_map = auto_map_columns(list(df_raw.columns))
    feature_cols = [col_map[p] for p in POLLUTANT_ORDER if col_map.get(p) in df_raw.columns]
    target_col = "ispu" if "ispu" in df_raw.columns else col_map.get("ispu")

    if not target_col or target_col not in df_raw.columns or len(feature_cols) < 2:
        raise ValueError(
            "Dataset bawaan tidak memiliki minimal 2 kolom polutan dan kolom target ISPU ('ispu' atau 'max')."
        )

    max_missing_frac = float(df_raw[feature_cols].isna().mean().max())
    missing_strategy = "drop" if max_missing_frac > 0.20 else "median"

    df_clean, clean_report = validate_and_clean_dataframe(
        df_raw,
        feature_cols=feature_cols,
        target_col=target_col,
        missing_strategy=missing_strategy,
        clip_negative=True,
    )
    trained = execute_training_pipeline(
        df_clean=df_clean,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=0.2,
        partition_type="grid",
        n_mfs=2,
        n_rules=12,
        mf_type="gaussian",
        epochs=40,
        learning_rate=0.02,
        l2_reg=1e-3,
        random_state=42,
    )
    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "clean_report": clean_report,
        "data_source_name": f"Dataset Bawaan ({DEFAULT_DATASET_PATH.name})",
        **trained,
    }


def apply_bundle_to_session(bundle: dict[str, Any]) -> None:
    """Menyimpan hasil pipeline ke st.session_state serta menyinkronkan input manual."""
    prev_mode = st.session_state.get("input_mode")
    for key, val in bundle.items():
        st.session_state[key] = val

    input_mode = bundle.get("input_mode", st.session_state.get("input_mode", "concentration"))
    mode_changed = prev_mode is not None and prev_mode != input_mode

    feature_cols: list[str] = bundle["split_data"]["feature_cols"]
    current_manual = st.session_state.get("manual_inputs", {})
    synced_manual: dict[str, float] = {}
    X_train_raw = bundle["split_data"]["X_train_raw"]
    for j, col in enumerate(feature_cols):
        if not mode_changed and col in current_manual:
            val = float(current_manual[col])
        elif input_mode == "sub_index" or col.lower() not in POLLUTANT_SPECS:
            val = round(float(np.median(X_train_raw[:, j])), 1)
        else:
            val = float(POLLUTANT_SPECS[col.lower()]["default_value"])
        synced_manual[col] = val
        if mode_changed:
            st.session_state[f"manual_field_{col}"] = val
    st.session_state["manual_inputs"] = synced_manual


def initialize_session_state() -> None:
    """Memastikan seluruh kunci session_state tersedia sejak halaman pertama dimuat."""
    if "anfis_model" not in st.session_state:
        bundle = build_initial_bundle()
        apply_bundle_to_session(bundle)
        st.session_state.setdefault("last_manual_result", None)
        st.session_state.setdefault("active_page", "1. Beranda")


def run_manual_prediction(raw_values: dict[str, Any]) -> dict[str, Any]:
    """Memvalidasi masukan manual dan menjalankan inferensi ANFIS + pembanding KLHK resmi."""
    split_data = st.session_state["split_data"]
    model: ANFIS = st.session_state["anfis_model"]
    input_mode: str = st.session_state.get("input_mode", "concentration")
    feature_cols: list[str] = split_data["feature_cols"]
    feature_bounds: dict[str, tuple[float, float]] = split_data["feature_bounds"]
    scaler = split_data["scaler"]

    ordered_inputs = {col: raw_values.get(col) for col in feature_cols}
    errors, warnings = validate_manual_input(
        ordered_inputs,
        feature_bounds=feature_bounds,
        input_mode=input_mode,
    )
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    x_raw = np.array([float(ordered_inputs[col]) for col in feature_cols], dtype=np.float64)
    x_scaled = scaler.transform(x_raw)
    pred_score = float(model.predict(x_scaled))

    official_vals = {
        col.lower(): float(ordered_inputs[col])
        for col in feature_cols
        if col.lower() in POLLUTANT_SPECS
    }
    if official_vals:
        official = (
            compute_ispu_from_subindices(official_vals)
            if input_mode == "sub_index"
            else compute_ispu_official(official_vals)
        )
    else:
        official = None

    result = {
        "ok": True,
        "errors": [],
        "warnings": warnings,
        "inputs": {col: float(ordered_inputs[col]) for col in feature_cols},
        "x_raw": x_raw,
        "x_scaled": x_scaled,
        "pred_score": pred_score,
        "official": official,
    }
    st.session_state["manual_inputs"] = result["inputs"]
    st.session_state["last_manual_result"] = result
    return result
