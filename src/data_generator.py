"""Generator dataset kualitas udara ISPU Indonesia yang deterministik dan realistis.

Menghasilkan `data/ispu_indonesia.csv` berisi 1.200 observasi harian polutan udara
(PM10, PM2.5, SO2, CO, O3, NO2) di 5 stasiun pemantau (SPKU) dengan skor ISPU,
kategori resmi Permen LHK No. 14 Tahun 2020, dan polutan kritis.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Pastikan impor berfungsi baik saat dijalankan sebagai skrip langsung maupun modul
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ispu_standard import POLLUTANT_ORDER, classify_ispu, compute_ispu_official

STATIONS: list[str] = [
    "SPKU-1 Pusat Kota",
    "SPKU-2 Kawasan Industri",
    "SPKU-3 Permukiman",
    "SPKU-4 Pinggiran Kota",
    "SPKU-5 Koridor Transportasi",
]

# Proporsi rezim kualitas udara agar seluruh 5 kategori ISPU terwakili seimbang
REGIMES: list[dict[str, object]] = [
    {
        "name": "Baik",
        "weight": 0.22,
        "means": {"pm10": 26.0, "pm25": 9.8, "so2": 22.0, "co": 1500.0, "o3": 58.0, "no2": 34.0},
        "stds": {"pm10": 9.0, "pm25": 3.2, "so2": 10.0, "co": 650.0, "o3": 22.0, "no2": 14.0},
    },
    {
        "name": "Sedang",
        "weight": 0.34,
        "means": {"pm10": 82.0, "pm25": 34.0, "so2": 82.0, "co": 5200.0, "o3": 148.0, "no2": 118.0},
        "stds": {"pm10": 24.0, "pm25": 10.5, "so2": 28.0, "co": 1200.0, "o3": 36.0, "no2": 32.0},
    },
    {
        "name": "Tidak Sehat",
        "weight": 0.23,
        "means": {"pm10": 205.0, "pm25": 92.0, "so2": 245.0, "co": 10200.0, "o3": 275.0, "no2": 460.0},
        "stds": {"pm10": 55.0, "pm25": 26.0, "so2": 65.0, "co": 2100.0, "o3": 55.0, "no2": 180.0},
    },
    {
        "name": "Sangat Tidak Sehat",
        "weight": 0.13,
        "means": {"pm10": 372.0, "pm25": 192.0, "so2": 530.0, "co": 21000.0, "o3": 510.0, "no2": 1350.0},
        "stds": {"pm10": 32.0, "pm25": 32.0, "so2": 120.0, "co": 4200.0, "o3": 120.0, "no2": 350.0},
    },
    {
        "name": "Berbahaya",
        "weight": 0.08,
        "means": {"pm10": 455.0, "pm25": 335.0, "so2": 910.0, "co": 34500.0, "o3": 810.0, "no2": 2350.0},
        "stds": {"pm10": 40.0, "pm25": 65.0, "so2": 150.0, "co": 5500.0, "o3": 115.0, "no2": 360.0},
    },
]


def generate_ispu_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Membuat dataset sintetis polusi udara Indonesia berbasis Permen LHK No. 14/2020."""
    rng = np.random.default_rng(seed)

    weights = np.array([float(r["weight"]) for r in REGIMES])
    weights = weights / weights.sum()
    regime_indices = rng.choice(len(REGIMES), size=n_samples, p=weights)

    start_date = date(2023, 1, 1)
    records: list[dict[str, object]] = []

    for idx in range(n_samples):
        regime = REGIMES[int(regime_indices[idx])]
        means = regime["means"]
        stds = regime["stds"]

        # Faktor latensi meteorologi (inversi suhu & kelembapan) untuk korelasi antarpolutan
        meteo_factor = float(rng.normal(0.0, 0.35))
        station_idx = idx % len(STATIONS)
        station = STATIONS[station_idx]

        # Pengaruh karakteristik lokasi SPKU (industri lebih tinggi SO2, transportasi lebih tinggi CO/NO2)
        station_bias = {
            0: {"pm10": 1.03, "pm25": 1.04, "so2": 0.95, "co": 1.06, "o3": 1.02, "no2": 1.05},
            1: {"pm10": 1.06, "pm25": 1.05, "so2": 1.15, "co": 1.00, "o3": 0.98, "no2": 1.08},
            2: {"pm10": 0.94, "pm25": 0.96, "so2": 0.90, "co": 0.92, "o3": 1.04, "no2": 0.90},
            3: {"pm10": 0.88, "pm25": 0.89, "so2": 0.85, "co": 0.86, "o3": 1.06, "no2": 0.84},
            4: {"pm10": 1.05, "pm25": 1.03, "so2": 0.98, "co": 1.14, "o3": 0.96, "no2": 1.15},
        }[station_idx]

        sample_vals: dict[str, float] = {}
        for pol in POLLUTANT_ORDER:
            raw = means[pol] + stds[pol] * (0.55 * meteo_factor + 0.835 * float(rng.normal()))
            raw *= station_bias[pol]
            min_floor = 2.0 if pol != "co" else 150.0
            sample_vals[pol] = round(float(max(min_floor, raw)), 2)

        # Pastikan konsistensi fisik: PM2.5 umumnya tidak melebihi PM10 secara ekstrem
        if sample_vals["pm25"] > sample_vals["pm10"] * 0.96:
            sample_vals["pm25"] = round(sample_vals["pm10"] * float(rng.uniform(0.65, 0.92)), 2)

        official = compute_ispu_official(sample_vals)

        # Tambahkan interaksi non-linear kecil antarpolutan (efek sinergis PM2.5-O3-NO2) + derau sensor
        synergy = (
            0.035 * np.sqrt(sample_vals["pm25"] * sample_vals["o3"])
            + 0.008 * np.sqrt(sample_vals["so2"] * sample_vals["no2"])
            - 2.5
        )
        noise = float(rng.normal(0.0, 1.8))
        ispu_score = round(float(np.clip(official["ispu"] + synergy + noise, 5.0, 500.0)), 2)
        cat_info = classify_ispu(ispu_score)

        obs_date = start_date + timedelta(days=idx // len(STATIONS))
        records.append(
            {
                "tanggal": obs_date.isoformat(),
                "stasiun": station,
                "pm10": sample_vals["pm10"],
                "pm25": sample_vals["pm25"],
                "so2": sample_vals["so2"],
                "co": sample_vals["co"],
                "o3": sample_vals["o3"],
                "no2": sample_vals["no2"],
                "ispu": ispu_score,
                "kategori": cat_info["name"],
                "polutan_kritis": official["polutan_kritis"],
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    output_path = PROJECT_ROOT / "data" / "ispu_indonesia.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_ispu_dataset(n_samples=1200, seed=42)
    df.to_csv(output_path, index=False)

    print(f"Dataset berhasil dibuat: {output_path} ({len(df)} baris)")
    print("\nDistribusi Kategori ISPU:")
    print(df["kategori"].value_counts())


if __name__ == "__main__":
    main()
