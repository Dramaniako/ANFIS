"""Contoh perhitungan manual ANFIS 5-Layer (2 Input x 2 MF = 4 Aturan).

Modul ini menyediakan contoh numerik ringkas dan terverifikasi secara programatik
terhadap mesin `ANFIS` sehingga mahasiswa dapat menjelaskan alur perhitungan
Lapisan 1 sampai Lapisan 5 secara eksak dalam laporan maupun presentasi.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.anfis import ANFIS


def compute_worked_manual_example() -> dict[str, Any]:
    """Menghitung contoh numerik 2-input (PM2.5, PM10) -> ISPU dan memverifikasinya via kelas ANFIS."""
    # Data mentah sampel uji beserta batas normalisasi Min-Max
    pm25_raw, pm25_min, pm25_max = 85.0, 5.0, 205.0
    pm10_raw, pm10_min, pm10_max = 180.0, 20.0, 340.0

    x1 = (pm25_raw - pm25_min) / (pm25_max - pm25_min)  # 80 / 200 = 0.40
    x2 = (pm10_raw - pm10_min) / (pm10_max - pm10_min)  # 160 / 320 = 0.50

    # Parameter premis Gaussian (c, sigma) untuk 2 MF per input (Rendah, Tinggi)
    c = np.array([[0.20, 0.80], [0.20, 0.80]], dtype=np.float64)
    sigma = np.array([[0.30, 0.30], [0.30, 0.30]], dtype=np.float64)

    # Layer 1: Fuzzifikasi
    mu_x1_low = float(np.exp(-0.5 * ((x1 - c[0, 0]) / sigma[0, 0]) ** 2))
    mu_x1_high = float(np.exp(-0.5 * ((x1 - c[0, 1]) / sigma[0, 1]) ** 2))
    mu_x2_low = float(np.exp(-0.5 * ((x2 - c[1, 0]) / sigma[1, 0]) ** 2))
    mu_x2_high = float(np.exp(-0.5 * ((x2 - c[1, 1]) / sigma[1, 1]) ** 2))

    # Layer 2: Firing Strength (Product T-Norm) untuk 4 aturan
    w1 = mu_x1_low * mu_x2_low
    w2 = mu_x1_low * mu_x2_high
    w3 = mu_x1_high * mu_x2_low
    w4 = mu_x1_high * mu_x2_high
    w_vec = np.array([w1, w2, w3, w4], dtype=np.float64)
    w_sum = float(np.sum(w_vec))

    # Layer 3: Normalisasi Firing Strength
    w_bar = w_vec / w_sum

    # Layer 4: Parameter konsekuen Takagi-Sugeno Orde-1 [p1, p2, r]
    consequents = np.array(
        [
            [20.0, 15.0, 25.0],
            [30.0, 95.0, 50.0],
            [110.0, 35.0, 65.0],
            [160.0, 140.0, 120.0],
        ],
        dtype=np.float64,
    )
    f_vals = np.array([p[0] * x1 + p[1] * x2 + p[2] for p in consequents], dtype=np.float64)
    weighted_f = w_bar * f_vals

    # Layer 5: Agregasi Output
    y_manual = float(np.sum(weighted_f))

    # Verifikasi menggunakan objek kelas ANFIS sesungguhnya
    engine = ANFIS(
        n_inputs=2,
        partition_type="grid",
        n_mfs=2,
        mf_type="gaussian",
        feature_names=["PM2.5", "PM10"],
    )
    engine.premise_params_ = {"c": c.copy(), "sigma": sigma.copy()}
    engine.initial_premise_params_ = {"c": c.copy(), "sigma": sigma.copy()}
    engine.rule_mf_indices_ = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    engine.consequent_params_ = consequents.copy()
    engine.is_fitted_ = True

    y_engine = float(engine.predict(np.array([x1, x2], dtype=np.float64)))

    rule_labels = [
        ("R1", "PM2.5 Rendah AND PM10 Rendah", "20·x1 + 15·x2 + 25"),
        ("R2", "PM2.5 Rendah AND PM10 Tinggi", "30·x1 + 95·x2 + 50"),
        ("R3", "PM2.5 Tinggi AND PM10 Rendah", "110·x1 + 35·x2 + 65"),
        ("R4", "PM2.5 Tinggi AND PM10 Tinggi", "160·x1 + 140·x2 + 120"),
    ]
    steps_df = pd.DataFrame(
        [
            {
                "Aturan": rule_labels[i][0],
                "Premis": rule_labels[i][1],
                "Fungsi f_k": rule_labels[i][2],
                "w_k (Layer 2)": float(w_vec[i]),
                "w_bar_k (Layer 3)": float(w_bar[i]),
                "f_k (Layer 4)": float(f_vals[i]),
                "w_bar_k * f_k": float(weighted_f[i]),
            }
            for i in range(4)
        ]
    )

    return {
        "raw_inputs": {"PM2.5": pm25_raw, "PM10": pm10_raw},
        "bounds": {"PM2.5": (pm25_min, pm25_max), "PM10": (pm10_min, pm10_max)},
        "x_norm": {"x1": x1, "x2": x2},
        "premise": {"c": c, "sigma": sigma},
        "layer1": {
            "mu_x1_low": mu_x1_low,
            "mu_x1_high": mu_x1_high,
            "mu_x2_low": mu_x2_low,
            "mu_x2_high": mu_x2_high,
        },
        "w": w_vec,
        "w_sum": w_sum,
        "w_bar": w_bar,
        "consequents": consequents,
        "f": f_vals,
        "weighted_f": weighted_f,
        "y_manual": y_manual,
        "y_engine": y_engine,
        "steps_df": steps_df,
    }
