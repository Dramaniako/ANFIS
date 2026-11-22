"""Standar Indeks Standar Pencemar Udara (ISPU) Indonesia.

Implementasi acuan resmi Peraturan Menteri Lingkungan Hidup dan Kehutanan (Permen LHK)
Republik Indonesia Nomor P.14/MENLHK/SETJEN/KUM.1/7/2020 tentang Indeks Standar
Pencemar Udara (ISPU), beserta validasi masukan dan skenario uji preset.
"""

from __future__ import annotations

import math
from typing import Any

# Breakpoint nilai indeks ISPU resmi (Lampiran I Permen LHK No. 14 Tahun 2020)
ISPU_BREAKPOINTS: list[float] = [0.0, 50.0, 100.0, 200.0, 300.0, 500.0]

POLLUTANT_ORDER: list[str] = ["pm10", "pm25", "so2", "co", "o3", "no2"]

POLLUTANT_SPECS: dict[str, dict[str, Any]] = {
    "pm10": {
        "name": "PM10",
        "full_name": "Partikulat Matter ≤ 10 µm",
        "unit": "µg/m³",
        "breakpoints": [0.0, 50.0, 150.0, 350.0, 420.0, 500.0],
        "max_physical": 1000.0,
        "default_value": 58.0,
        "description": "Partikel debu halus berdiameter ≤ 10 mikrometer dari jalan raya, konstruksi, dan pembakaran.",
        "health_impact": "Iritasi saluran pernapasan atas, batuk, dan memperburuk asma.",
    },
    "pm25": {
        "name": "PM2.5",
        "full_name": "Partikulat Matter ≤ 2,5 µm",
        "unit": "µg/m³",
        "breakpoints": [0.0, 15.5, 55.4, 150.4, 250.4, 500.0],
        "max_physical": 1000.0,
        "default_value": 32.0,
        "description": "Partikel ultra-halus berdiameter ≤ 2,5 mikrometer yang mampu menembus alveolus paru.",
        "health_impact": "Risiko penyakit kardiovaskular, bronkitis, dan penurunan fungsi paru.",
    },
    "so2": {
        "name": "SO2",
        "full_name": "Sulfur Dioksida",
        "unit": "µg/m³",
        "breakpoints": [0.0, 52.0, 180.0, 400.0, 800.0, 1200.0],
        "max_physical": 2500.0,
        "default_value": 45.0,
        "description": "Gas hasil pembakaran bahan bakar fosil berkadar sulfur tinggi (industri & PLTU).",
        "health_impact": "Bronkokonstriksi, sesak napas, dan pemicu hujan asam.",
    },
    "co": {
        "name": "CO",
        "full_name": "Karbon Monoksida",
        "unit": "µg/m³",
        "breakpoints": [0.0, 4000.0, 8000.0, 15000.0, 30000.0, 45000.0],
        "max_physical": 90000.0,
        "default_value": 3200.0,
        "description": "Gas beracun tidak berwarna akibat pembakaran tidak sempurna kendaraan bermotor.",
        "health_impact": "Mengikat hemoglobin darah dan menghambat suplai oksigen ke jaringan tubuh.",
    },
    "o3": {
        "name": "O3",
        "full_name": "Ozon Permukaan (Troposfer)",
        "unit": "µg/m³",
        "breakpoints": [0.0, 120.0, 235.0, 400.0, 800.0, 1000.0],
        "max_physical": 2000.0,
        "default_value": 95.0,
        "description": "Polutan sekunder hasil reaksi fotokimia NOx dan VOC di bawah sinar matahari.",
        "health_impact": "Radang jaringan paru, nyeri dada saat bernapas dalam, dan iritasi mata.",
    },
    "no2": {
        "name": "NO2",
        "full_name": "Nitrogen Dioksida",
        "unit": "µg/m³",
        "breakpoints": [0.0, 80.0, 200.0, 1130.0, 2260.0, 3000.0],
        "max_physical": 5000.0,
        "default_value": 75.0,
        "description": "Gas kecokelatan yang bersumber terutama dari emisi kendaraan diesel dan pembangkit listrik.",
        "health_impact": "Peradangan saluran napas dan peningkatan kerentanan infeksi pernapasan.",
    },
}

ISPU_CATEGORIES: list[dict[str, Any]] = [
    {
        "name": "Baik",
        "min_val": 0.0,
        "max_val": 50.0,
        "range_label": "0 – 50",
        "color": "#16a34a",
        "bg_light": "#dcfce7",
        "text_color": "#ffffff",
        "badge_label": "Hijau",
        "recommendation": (
            "Tingkat kualitas udara sangat baik dan tidak memberikan efek negatif "
            "terhadap kesehatan manusia, hewan, maupun tumbuhan. Aktivitas luar ruangan dapat dilakukan secara normal."
        ),
    },
    {
        "name": "Sedang",
        "min_val": 50.0,
        "max_val": 100.0,
        "range_label": "51 – 100",
        "color": "#2563eb",
        "bg_light": "#dbeafe",
        "text_color": "#ffffff",
        "badge_label": "Biru",
        "recommendation": (
            "Kualitas udara masih dapat diterima pada kesehatan manusia, namun kelompok sangat sensitif "
            "(penderita asma, anak-anak, lansia) sebaiknya mengurangi aktivitas fisik berat di luar ruangan."
        ),
    },
    {
        "name": "Tidak Sehat",
        "min_val": 100.0,
        "max_val": 200.0,
        "range_label": "101 – 200",
        "color": "#d97706",
        "bg_light": "#fef3c7",
        "text_color": "#ffffff",
        "badge_label": "Kuning",
        "recommendation": (
            "Bersifat merugikan bagi manusia, hewan, dan tumbuhan sensitif. Gunakan masker respirator (N95/KF94) "
            "saat beraktivitas di luar dan kurangi olahraga luar ruangan berkepanjangan."
        ),
    },
    {
        "name": "Sangat Tidak Sehat",
        "min_val": 200.0,
        "max_val": 300.0,
        "range_label": "201 – 300",
        "color": "#dc2626",
        "bg_light": "#fee2e2",
        "text_color": "#ffffff",
        "badge_label": "Merah",
        "recommendation": (
            "Kualitas udara dapat meningkatkan risiko gangguan pernapasan serius pada populasi umum. "
            "Hindari aktivitas luar ruangan, tutup ventilasi rumah, dan aktifkan penyaring udara (air purifier)."
        ),
    },
    {
        "name": "Berbahaya",
        "min_val": 300.0,
        "max_val": float("inf"),
        "range_label": "> 300",
        "color": "#1f2937",
        "bg_light": "#e5e7eb",
        "text_color": "#ffffff",
        "badge_label": "Hitam",
        "recommendation": (
            "Kondisi darurat polusi udara yang berdampak serius bagi seluruh populasi. Tetap berada di dalam ruangan "
            "tertutup, gunakan masker pelindung tingkat tinggi bila terpaksa keluar, dan pantau imbauan pemerintah."
        ),
    },
]

PRESET_SCENARIOS: dict[str, dict[str, Any]] = {
    "Pagi Cerah / Udara Bersih": {
        "description": "Kondisi pagi hari di kawasan permukiman hijau dengan sirkulasi angin baik (target kategori Baik).",
        "values": {"pm10": 28.0, "pm25": 10.5, "so2": 22.0, "co": 1400.0, "o3": 55.0, "no2": 35.0},
    },
    "Jam Sibuk Perkotaan": {
        "description": "Koridor lalu lintas padat saat jam berangkat/pulang kerja (target kategori Sedang).",
        "values": {"pm10": 88.0, "pm25": 36.0, "so2": 72.0, "co": 5800.0, "o3": 145.0, "no2": 135.0},
    },
    "Polusi Kawasan Industri": {
        "description": "Paparan emisi industri berat dan transportasi logistik (target kategori Tidak Sehat).",
        "values": {"pm10": 215.0, "pm25": 98.0, "so2": 265.0, "co": 10800.0, "o3": 295.0, "no2": 480.0},
    },
    "Kabut Asap Ekstrem": {
        "description": "Episode kabut asap kebakaran lahan / stagnasi atmosfer parah (target kategori Sangat Tidak Sehat / Berbahaya).",
        "values": {"pm10": 435.0, "pm25": 285.0, "so2": 620.0, "co": 28500.0, "o3": 560.0, "no2": 1520.0},
    },
}

PRESET_SCENARIOS_SUBINDEX: dict[str, dict[str, Any]] = {
    "Pagi Cerah / Udara Bersih": {
        "description": "Kondisi pagi hari Jakarta dengan seluruh sub-indeks polutan di bawah 50 (target kategori Baik).",
        "values": {"pm10": 26.0, "pm25": 38.0, "so2": 22.0, "co": 9.0, "o3": 24.0, "no2": 12.0},
    },
    "Jam Sibuk Perkotaan": {
        "description": "Kondisi lalu lintas harian Jakarta dengan dominasi PM2.5 pada rentang Sedang (51–100).",
        "values": {"pm10": 55.0, "pm25": 84.0, "so2": 41.0, "co": 15.0, "o3": 36.0, "no2": 24.0},
    },
    "Polusi Tinggi Perkotaan": {
        "description": "Episode pencemaran perkotaan dengan sub-indeks PM2.5 dan O3 pada kategori Tidak Sehat (101–200).",
        "values": {"pm10": 92.0, "pm25": 148.0, "so2": 58.0, "co": 28.0, "o3": 112.0, "no2": 48.0},
    },
    "Polusi Ekstrem / Kabut Asap": {
        "description": "Episode pencemaran berat Jakarta dengan sub-indeks di atas 200 (target Sangat Tidak Sehat).",
        "values": {"pm10": 145.0, "pm25": 225.0, "so2": 78.0, "co": 42.0, "o3": 155.0, "no2": 85.0},
    },
}


def compute_sub_index(value: float, pollutant: str) -> float:
    """Menghitung sub-indeks ISPU untuk satu parameter polutan sesuai Permen LHK No. 14/2020.

    Menggunakan interpolasi linear bertingkat:
        I = ((Ia - Ib) / (Xa - Xb)) * (Xx - Xb) + Ib
    """
    key = pollutant.lower()
    if key not in POLLUTANT_SPECS:
        raise ValueError(f"Parameter polutan tidak dikenal: {pollutant}")

    x_val = float(value)
    if math.isnan(x_val) or math.isinf(x_val):
        raise ValueError(f"Nilai konsentrasi {pollutant} tidak valid: {value}")
    if x_val <= 0.0:
        return 0.0

    x_bps = POLLUTANT_SPECS[key]["breakpoints"]
    i_bps = ISPU_BREAKPOINTS

    for idx in range(len(x_bps) - 1):
        xb, xa = x_bps[idx], x_bps[idx + 1]
        ib, ia = i_bps[idx], i_bps[idx + 1]
        if x_val <= xa:
            return ((ia - ib) / (xa - xb)) * (x_val - xb) + ib

    # Ekstrapolasi linear apabila konsentrasi melebihi batas atas segmen 500
    xb, xa = x_bps[-2], x_bps[-1]
    ib, ia = i_bps[-2], i_bps[-1]
    return ((ia - ib) / (xa - xb)) * (x_val - xb) + ib


def classify_ispu(score: float) -> dict[str, Any]:
    """Mengklasifikasikan nilai skor ISPU kontinu ke dalam 5 kategori resmi Permen LHK 14/2020."""
    val = float(score)
    if val <= 50.0:
        return dict(ISPU_CATEGORIES[0])
    if val <= 100.0:
        return dict(ISPU_CATEGORIES[1])
    if val <= 200.0:
        return dict(ISPU_CATEGORIES[2])
    if val <= 300.0:
        return dict(ISPU_CATEGORIES[3])
    return dict(ISPU_CATEGORIES[4])


def _build_ispu_result(sub_indices: dict[str, float]) -> dict[str, Any]:
    """Menyusun hasil ringkasan ISPU dari kamus nilai sub-indeks per polutan."""
    if not sub_indices:
        raise ValueError("Minimal satu parameter polutan resmi harus disediakan untuk menghitung ISPU.")

    critical_key = max(sub_indices, key=lambda k: sub_indices[k])
    ispu_val = sub_indices[critical_key]
    cat_info = classify_ispu(ispu_val)

    return {
        "ispu": float(ispu_val),
        "kategori": cat_info["name"],
        "kategori_info": cat_info,
        "polutan_kritis": POLLUTANT_SPECS[critical_key]["name"],
        "polutan_kritis_key": critical_key,
        "sub_indices": sub_indices,
    }


def compute_ispu_official(values: dict[str, float]) -> dict[str, Any]:
    """Menghitung nilai ISPU resmi dari konsentrasi mentah (µg/m³) per polutan."""
    sub_indices: dict[str, float] = {}
    for key in POLLUTANT_ORDER:
        if key in values and values[key] is not None:
            sub_indices[key] = compute_sub_index(float(values[key]), key)
    return _build_ispu_result(sub_indices)


def compute_ispu_from_subindices(values: dict[str, float]) -> dict[str, Any]:
    """Menghitung nilai ISPU resmi apabila masukan sudah berupa sub-indeks per polutan."""
    sub_indices: dict[str, float] = {}
    for key in POLLUTANT_ORDER:
        if key in values and values[key] is not None:
            sub_indices[key] = float(values[key])
    return _build_ispu_result(sub_indices)


def validate_manual_input(
    values: dict[str, Any],
    feature_bounds: dict[str, tuple[float, float]] | None = None,
    input_mode: str = "concentration",
) -> tuple[list[str], list[str]]:
    """Memvalidasi masukan manual pengguna sebelum inferensi ANFIS.

    Mengembalikan tuple (errors, warnings):
    - errors: kesalahan fatal yang menghalangi proses prediksi.
    - warnings: peringatan ekstrapolasi apabila nilai di luar rentang data latih.
    """
    errors: list[str] = []
    warnings: list[str] = []
    is_subindex = input_mode == "sub_index"

    for col, raw_val in values.items():
        spec = POLLUTANT_SPECS.get(col.lower())
        label = spec["name"] if spec else col
        unit = "Sub-Indeks ISPU" if is_subindex else (spec["unit"] if spec else "")
        noun = "Sub-indeks" if is_subindex else "Konsentrasi"

        if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
            errors.append(f"Nilai {label} tidak boleh kosong.")
            continue

        try:
            num = float(raw_val)
        except (TypeError, ValueError):
            errors.append(f"Nilai {label} harus berupa angka valid (diterima: {raw_val!r}).")
            continue

        if math.isnan(num) or math.isinf(num):
            errors.append(f"Nilai {label} tidak boleh NaN atau tak hingga.")
            continue

        if num < 0.0:
            errors.append(f"{noun} {label} tidak boleh bernilai negatif ({num:.2f}).")
            continue

        max_ceiling = 1000.0 if is_subindex else (spec["max_physical"] if spec is not None else None)
        if max_ceiling is not None and num > max_ceiling:
            errors.append(
                f"{noun} {label} ({num:.2f} {unit}) melebihi batas fisik wajar ({max_ceiling:.0f} {unit})."
            )
            continue

        if feature_bounds and col in feature_bounds:
            low, high = feature_bounds[col]
            if num < low or num > high:
                warnings.append(
                    f"Nilai {label} = {num:.2f} {unit} berada di luar rentang data latih "
                    f"[{low:.2f}, {high:.2f}] sehingga prediksi bersifat ekstrapolasi."
                )

    return errors, warnings
