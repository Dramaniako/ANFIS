"""Modul prapemrosesan, pembersihan, validasi, dan normalisasi data polusi udara."""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ispu_standard import POLLUTANT_ORDER, classify_ispu

_PLACEHOLDER_STRINGS = {"", "-", "--", "---", "na", "n/a", "null", "none", "nan", "?"}

_COLUMN_ALIASES: dict[str, list[str]] = {
    "pm10": ["pm10", "pm_10", "pm 10", "partikulat10"],
    "pm25": ["pm25", "pm2.5", "pm_25", "pm2_5", "pm 2.5", "pm 25", "partikulat25"],
    "so2": ["so2", "so_2", "so 2", "sulfur_dioksida", "sulfurdioxide"],
    "co": ["co", "karbon_monoksida", "carbon_monoxide"],
    "o3": ["o3", "o_3", "o 3", "ozon", "ozone"],
    "no2": ["no2", "no_2", "no 2", "nitrogen_dioksida", "nitrogendioxide"],
    "ispu": ["ispu", "nilai_ispu", "skor_ispu", "max", "aqi", "score", "target", "indeks"],
}


class MinMaxScalerCustom:
    """Normalisasi Min-Max murni NumPy ke rentang [0, 1] dengan pengaman kolom konstan.

    Rumus:
        x_scaled = (x - x_min) / (x_max - x_min)
    """

    def __init__(self) -> None:
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> MinMaxScalerCustom:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Input X untuk MinMaxScalerCustom harus berupa array 2D.")
        self.data_min_ = np.min(arr, axis=0)
        self.data_max_ = np.max(arr, axis=0)
        raw_span = self.data_max_ - self.data_min_
        self.data_range_ = np.where(raw_span < 1e-12, 1.0, raw_span)
        self.feature_names_ = list(feature_names) if feature_names else [f"x{i + 1}" for i in range(arr.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError("Scaler belum di-fit pada data latih.")
        arr = np.asarray(X, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        scaled = (arr - self.data_min_) / self.data_range_
        return scaled[0] if single else scaled

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError("Scaler belum di-fit pada data latih.")
        arr = np.asarray(X_scaled, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        restored = arr * self.data_range_ + self.data_min_
        return restored[0] if single else restored

    def fit_transform(self, X: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        return self.fit(X, feature_names=feature_names).transform(X)


def _normalize_col_token(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(name).strip().lower())


def auto_map_columns(columns: list[str]) -> dict[str, str | None]:
    """Memetakan nama kolom CSV secara otomatis ke variabel standar ISPU."""
    normalized_lookup = {_normalize_col_token(c): c for c in columns}
    mapping: dict[str, str | None] = {}

    for std_key in [*POLLUTANT_ORDER, "ispu"]:
        matched: str | None = None
        aliases = _COLUMN_ALIASES.get(std_key, [std_key])
        for alias in aliases:
            tok = _normalize_col_token(alias)
            if tok in normalized_lookup:
                matched = normalized_lookup[tok]
                break
        if matched is None:
            # Pencocokan substring ringan sebagai cadangan
            for col in columns:
                tok = _normalize_col_token(col)
                if std_key in tok:
                    matched = col
                    break
        mapping[std_key] = matched

    return mapping


def validate_and_clean_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    missing_strategy: str = "median",
    clip_negative: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Memvalidasi, mengonversi numerik, dan membersihkan dataset sebelum pelatihan."""
    if df is None or df.empty:
        raise ValueError("Dataset kosong. Harap unggah atau pilih dataset yang memiliki data.")

    if len(feature_cols) < 2:
        raise ValueError("Minimal 2 kolom fitur input harus dipilih untuk membangun aturan ANFIS.")

    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("Terdapat duplikasi pada kolom fitur input yang dipilih.")

    if target_col in feature_cols:
        raise ValueError(f"Kolom target '{target_col}' tidak boleh merangkap sebagai kolom fitur input.")

    missing_cols = [c for c in [*feature_cols, target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan pada dataset: {', '.join(missing_cols)}")

    work = df.copy()
    raw_rows = len(work)
    selected_cols = [*feature_cols, target_col]

    # Bersihkan string placeholder sensor seperti "-", "---", "NA", ""
    for col in selected_cols:
        if work[col].dtype == object or str(work[col].dtype) == "string":
            cleaned_str = work[col].astype(str).str.strip()
            mask_placeholder = cleaned_str.str.lower().isin(_PLACEHOLDER_STRINGS)
            work.loc[mask_placeholder, col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Hitung sel kosong setelah konversi numerik
    missing_target_count = int(work[target_col].isna().sum())
    missing_feature_cells = int(work[feature_cols].isna().sum().sum())

    # Baris dengan target kosong tidak dapat digunakan sebagai ground truth supervised learning
    work = work.dropna(subset=[target_col])

    imputed_cells = 0
    dropped_missing_rows = missing_target_count

    if missing_strategy == "drop":
        before = len(work)
        work = work.dropna(subset=feature_cols)
        dropped_missing_rows += before - len(work)
    elif missing_strategy == "median":
        for col in feature_cols:
            na_mask = work[col].isna()
            n_na = int(na_mask.sum())
            if n_na > 0:
                med = float(work[col].median())
                if np.isnan(med):
                    raise ValueError(f"Seluruh nilai pada kolom fitur '{col}' kosong/tidak valid.")
                work.loc[na_mask, col] = med
                imputed_cells += n_na
    else:
        raise ValueError(f"Strategi missing value tidak dikenal: {missing_strategy}")

    # Tangani nilai negatif konsentrasi / target
    negative_fixed = 0
    for col in selected_cols:
        neg_mask = work[col] < 0.0
        n_neg = int(neg_mask.sum())
        if n_neg > 0:
            if clip_negative:
                work.loc[neg_mask, col] = 0.0
                negative_fixed += n_neg
            else:
                work = work.loc[~neg_mask]
                negative_fixed += n_neg

    work = work.reset_index(drop=True)

    if len(work) < 25:
        raise ValueError(
            f"Jumlah baris valid setelah pembersihan hanya {len(work)} baris (minimal 25 baris diperlukan untuk split latih/uji ANFIS)."
        )

    # Ringkasan outlier IQR (informatif, tidak otomatis dibuang agar kasus ekstrem Berbahaya tetap terjaga)
    outlier_cells = 0
    for col in feature_cols:
        q1 = float(work[col].quantile(0.25))
        q3 = float(work[col].quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            outliers = (work[col] < (q1 - 1.5 * iqr)) | (work[col] > (q3 + 1.5 * iqr))
            outlier_cells += int(outliers.sum())

    report = {
        "raw_rows": raw_rows,
        "clean_rows": len(work),
        "dropped_rows": raw_rows - len(work),
        "missing_target_rows": missing_target_count,
        "missing_feature_cells": missing_feature_cells,
        "imputed_cells": imputed_cells,
        "negative_fixed": negative_fixed,
        "outlier_cells_iqr": outlier_cells,
        "missing_strategy": missing_strategy,
        "clip_negative": clip_negative,
    }
    return work, report


def prepare_train_test_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Membagi dataset latih/uji dan melakukan normalisasi Min-Max bebas kebocoran data."""
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("Rasio data uji harus berada pada rentang 0,05 hingga 0,50.")

    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    y_all = df[target_col].to_numpy(dtype=np.float64)
    indices = np.arange(len(df))

    # Stratifikasi berdasarkan kategori ISPU agar distribusi 5 kategori seimbang di latih & uji
    strata = pd.Series([classify_ispu(v)["name"] for v in y_all])
    can_stratify = bool((strata.value_counts() >= 2).all() and len(strata.unique()) > 1)
    stratify_arg = strata if can_stratify else None

    (
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        train_idx,
        test_idx,
    ) = train_test_split(
        X_raw,
        y_all,
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    # Fit scaler HANYA pada X_train untuk mencegah data leakage
    scaler = MinMaxScalerCustom()
    X_train = scaler.fit_transform(X_train_raw, feature_names=feature_cols)
    X_test = scaler.transform(X_test_raw)

    feature_bounds = {
        col: (float(scaler.data_min_[i]), float(scaler.data_max_[i]))
        for i, col in enumerate(feature_cols)
    }

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_indices": train_idx,
        "test_indices": test_idx,
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "feature_bounds": feature_bounds,
        "stratified": can_stratify,
    }


def detect_input_mode(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Literal["sub_index", "concentration"]:
    """Mendeteksi apakah kolom fitur sudah berupa sub-indeks ISPU atau konsentrasi mentah."""
    if df is None or df.empty or not feature_cols:
        return "concentration"

    row_max = df[feature_cols].max(axis=1)
    median_diff = float((df[target_col] - row_max).abs().median())

    co_col = next((c for c in feature_cols if c.strip().lower() == "co"), None)
    co_small = True if co_col is None else float(df[co_col].quantile(0.95)) <= 500.0

    if median_diff <= 5.0 and co_small:
        return "sub_index"
    return "concentration"
