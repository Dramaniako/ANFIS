"""Modul evaluasi metrik regresi, klasifikasi kategori ISPU, dan komparasi model baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

from src.ispu_standard import ISPU_CATEGORIES, classify_ispu

CATEGORY_LABELS: list[str] = [cat["name"] for cat in ISPU_CATEGORIES]


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Menghitung metrik evaluasi regresi: RMSE, MAE, MSE, R², dan MAPE."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()

    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0

    # Hindari pembagian nol pada MAPE
    safe_denom = np.where(np.abs(yt) < 1e-6, 1.0, np.abs(yt))
    mape = float(np.mean(np.abs((yt - yp) / safe_denom)) * 100.0)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "MAPE": mape,
    }


def compute_classification_metrics(y_true_scores: np.ndarray, y_pred_scores: np.ndarray) -> dict[str, Any]:
    """Mengonversi skor ISPU ke 5 kategori Permen LHK dan menghitung akurasi serta confusion matrix."""
    yt = np.asarray(y_true_scores, dtype=np.float64).ravel()
    yp = np.asarray(y_pred_scores, dtype=np.float64).ravel()

    true_cats = [classify_ispu(v)["name"] for v in yt]
    pred_cats = [classify_ispu(v)["name"] for v in yp]

    acc = float(accuracy_score(true_cats, pred_cats))
    macro_f1 = float(f1_score(true_cats, pred_cats, labels=CATEGORY_LABELS, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_cats, pred_cats, labels=CATEGORY_LABELS, average="weighted", zero_division=0))
    cm = confusion_matrix(true_cats, pred_cats, labels=CATEGORY_LABELS)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": list(CATEGORY_LABELS),
        "confusion_matrix": cm,
        "true_categories": true_cats,
        "pred_categories": pred_cats,
    }


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Melatih model pembanding (Linear Regression, MLP, Random Forest) pada split data yang sama."""
    candidates = [
        ("Regresi Linear Berganda", LinearRegression()),
        (
            "MLP Neural Network",
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                max_iter=600,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=1,
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    for name, est in candidates:
        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        y_pred_test = est.predict(X_test)

        reg = compute_regression_metrics(y_test, y_pred_test)
        cls = compute_classification_metrics(y_test, y_pred_test)
        results.append(
            {
                "Model": name,
                "RMSE": reg["RMSE"],
                "MAE": reg["MAE"],
                "R2": reg["R2"],
                "MAPE": reg["MAPE"],
                "Akurasi_Kategori": cls["accuracy"] * 100.0,
                "Macro_F1": cls["macro_f1"] * 100.0,
                "Waktu_Latih_s": elapsed,
                "y_pred_test": y_pred_test,
            }
        )
    return results


def generate_system_analysis(
    anfis_test_reg: dict[str, float],
    anfis_train_reg: dict[str, float],
    anfis_test_cls: dict[str, Any],
    baseline_results: list[dict[str, Any]],
    model_summary: dict[str, Any],
) -> dict[str, list[str]]:
    """Menyusun interpretasi ilmiah kualitas, kelebihan, dan keterbatasan model ANFIS."""
    r2_test = anfis_test_reg["R2"]
    rmse_test = anfis_test_reg["RMSE"]
    rmse_train = anfis_train_reg["RMSE"]
    acc_pct = anfis_test_cls["accuracy"] * 100.0

    lin_row = next((r for r in baseline_results if "Linear" in r["Model"]), None)

    kualitas: list[str] = []
    if r2_test >= 0.95:
        kualitas.append(
            f"Model ANFIS mencapai kinerja sangat baik dengan koefisien determinasi uji R² = {r2_test:.4f} "
            f"(RMSE uji = {rmse_test:.2f} poin ISPU) dan ketepatan kategori ISPU sebesar {acc_pct:.1f}%."
        )
    elif r2_test >= 0.85:
        kualitas.append(
            f"Model ANFIS menunjukkan akurasi prediksi baik dengan R² = {r2_test:.4f} "
            f"dan akurasi klasifikasi kategori ISPU {acc_pct:.1f}%."
        )
    else:
        kualitas.append(
            f"Model memperoleh R² uji = {r2_test:.4f} (RMSE = {rmse_test:.2f}); "
            "disarankan meninjau jumlah aturan/epoch atau kualitas data masukan."
        )

    gap = rmse_test - rmse_train
    if gap <= 3.0:
        kualitas.append(
            f"Selisih RMSE uji terhadap latih relatif kecil ({rmse_test:.2f} vs {rmse_train:.2f}), "
            "menunjukkan regulasi Ridge LSE berhasil menjaga generalisasi tanpa overfitting berlebih."
        )
    else:
        kualitas.append(
            f"Terdapat selisih RMSE uji terhadap latih ({rmse_test:.2f} vs {rmse_train:.2f}); "
            "pertimbangkan menaikkan regularisasi L2 (λ) atau memakai Scatter Partitioning agar aturan lebih ringkas."
        )

    if lin_row is not None and rmse_test < lin_row["RMSE"]:
        improv = ((lin_row["RMSE"] - rmse_test) / max(lin_row["RMSE"], 1e-9)) * 100.0
        kualitas.append(
            f"Dibandingkan Regresi Linear Berganda (RMSE = {lin_row['RMSE']:.2f}), ANFIS menurunkan kesalahan prediksi "
            f"sebesar {improv:.1f}% karena mampu memodelkan kurva piecewise-linear Permen LHK dan efek polutan dominan."
        )

    kelebihan = [
        "Interpretabilitas tinggi: setiap prediksi dapat ditelusuri langkah demi langkah melalui Lapisan 1–5 beserta bobot aturan IF-THEN aktif.",
        "Hybrid Learning (LSE + Backpropagation) mempercepat konvergensi parameter konsekuen linear secara eksak pada setiap epoch sambil menyempurnakan kurva fungsi keanggotaan.",
        "Cocok untuk data lingkungan polusi udara yang memiliki transisi ambang batas halus (fuzzy) antar kategori Baik, Sedang, hingga Berbahaya.",
    ]

    keterbatasan = [
        f"Mode {model_summary['partition_type'].title()} saat ini menggunakan {model_summary['n_rules']} aturan dan "
        f"{model_summary['total_params']} parameter; pada Grid Partitioning, jumlah aturan bertumbuh eksponensial (M^D) bila fitur ditambah.",
        "Akurasi ekstrapolasi menurun jika konsentrasi polutan uji berada jauh di luar rentang minimum–maksimum data latih.",
        "Model mempelajari bobot dari distribusi data historis sehingga belum memperhitungkan variabel meteorologi eksplisit (kecepatan angin, curah hujan, suhu) kecuali disertakan pada dataset unggahan.",
    ]

    return {
        "kualitas": kualitas,
        "kelebihan": kelebihan,
        "keterbatasan": keterbatasan,
    }
