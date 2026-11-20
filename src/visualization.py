"""Modul visualisasi interaktif berbasis Plotly untuk sistem AirFuzzy-ANFIS."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.anfis import ANFIS
from src.ispu_standard import ISPU_CATEGORIES, POLLUTANT_SPECS, classify_ispu
from src.preprocessing import MinMaxScalerCustom

_FONT_FAMILY = "Inter, Segoe UI, Arial, sans-serif"
_PALETTE = [
    "#0f766e",  # teal
    "#2563eb",  # blue
    "#d97706",  # amber
    "#dc2626",  # red
    "#7c3aed",  # violet
    "#059669",  # emerald
    "#0891b2",  # cyan
    "#db2777",  # pink
]


def _apply_base_layout(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0f172a", family=_FONT_FAMILY), x=0.01),
        template="plotly_white",
        height=height,
        font=dict(family=_FONT_FAMILY, size=12, color="#1e293b"),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=48, r=24, t=54, b=46),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1")
    return fig


def plot_anfis_architecture(
    n_inputs: int,
    feature_names: list[str],
    partition_type: str,
    n_mfs: int | None,
    n_rules: int,
) -> go.Figure:
    """Menggambar diagram skematik arsitektur jaringan 5-Lapisan ANFIS secara interaktif."""
    fig = go.Figure()

    layer_x = [0.0, 1.8, 3.6, 5.4, 7.2, 9.0]
    layer_titles = [
        "Input<br>(Layer 0)",
        "Layer 1<br>Fuzzifikasi<br>(Adaptif)",
        "Layer 2<br>Aturan Π<br>(Tetap)",
        "Layer 3<br>Normalisasi N<br>(Tetap)",
        "Layer 4<br>Konsekuen<br>(Adaptif)",
        "Layer 5<br>Agregasi Σ<br>(Tetap)",
    ]

    # Batasi node yang ditampilkan agar tetap rapi jika aturan > 8
    max_rules_shown = 8
    show_rules = min(n_rules, max_rules_shown)
    truncated = n_rules > max_rules_shown

    def y_positions(count: int, span: float = 5.6) -> list[float]:
        if count == 1:
            return [0.0]
        return list(np.linspace(span / 2.0, -span / 2.0, count))

    in_ys = y_positions(n_inputs, span=5.2)

    if partition_type == "grid" and n_mfs is not None:
        l1_count = min(n_inputs * n_mfs, 12)
        l1_ys = y_positions(l1_count, span=5.6)
        l1_labels = [
            f"μ({feature_names[i // n_mfs]},{(i % n_mfs) + 1})"
            for i in range(l1_count)
        ]
    else:
        l1_count = show_rules
        l1_ys = y_positions(l1_count, span=5.4)
        l1_labels = [f"μ_R{i + 1}" for i in range(l1_count)]

    rule_ys = y_positions(show_rules, span=5.4)
    out_ys = [0.0]

    # Gambar garis penghubung (edges) tipis terlebih dahulu
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    # Input -> Layer 1
    for i, yi in enumerate(in_ys):
        if partition_type == "grid" and n_mfs is not None:
            for m in range(n_mfs):
                l1_idx = i * n_mfs + m
                if l1_idx < l1_count:
                    edge_x.extend([layer_x[0], layer_x[1], None])
                    edge_y.extend([yi, l1_ys[l1_idx], None])
        else:
            for yl1 in l1_ys:
                edge_x.extend([layer_x[0], layer_x[1], None])
                edge_y.extend([yi, yl1, None])

    # Layer 1 -> Layer 2
    for yl1 in l1_ys:
        for yr in rule_ys:
            if partition_type == "grid" or abs(yl1 - yr) < 1e-6:
                edge_x.extend([layer_x[1], layer_x[2], None])
                edge_y.extend([yl1, yr, None])

    # Layer 2 -> Layer 3 -> Layer 4 -> Layer 5
    for yr in rule_ys:
        edge_x.extend([layer_x[2], layer_x[3], None])
        edge_y.extend([yr, yr, None])
        edge_x.extend([layer_x[3], layer_x[4], None])
        edge_y.extend([yr, yr, None])
        edge_x.extend([layer_x[4], layer_x[5], None])
        edge_y.extend([yr, out_ys[0], None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.45)", width=1.1),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Tambahkan node per lapisan (Kotak = Node Adaptif, Lingkaran = Node Tetap)
    def add_nodes(
        x_pos: float,
        ys: list[float],
        texts: list[str],
        hovers: list[str],
        name: str,
        color: str,
        symbol: str,
        size: int = 24,
        showlegend: bool = True,
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x_pos] * len(ys),
                y=ys,
                mode="markers+text",
                marker=dict(size=size, color=color, symbol=symbol, line=dict(width=1.5, color="#ffffff")),
                text=texts,
                textposition="middle center",
                textfont=dict(size=9, color="#ffffff", family=_FONT_FAMILY),
                hovertext=hovers,
                hoverinfo="text",
                name=name,
                showlegend=showlegend,
            )
        )

    in_texts = [fn[:5] for fn in feature_names]
    in_hovers = [f"Input: {fn}" for fn in feature_names]
    add_nodes(layer_x[0], in_ys, in_texts, in_hovers, "Input Variabel", "#475569", "circle", size=28)

    add_nodes(
        layer_x[1],
        l1_ys,
        ["A" for _ in l1_ys],
        l1_labels,
        "Node Adaptif (L1 & L4)",
        "#0f766e",
        "square",
        size=22,
    )
    add_nodes(
        layer_x[2],
        rule_ys,
        ["Π" for _ in rule_ys],
        [f"Layer 2 Firing Strength R{i + 1}" for i in range(show_rules)],
        "Node Tetap (L2, L3, L5)",
        "#2563eb",
        "circle",
        size=22,
    )
    add_nodes(
        layer_x[3],
        rule_ys,
        ["N" for _ in rule_ys],
        [f"Layer 3 Normalisasi w_bar{i + 1}" for i in range(show_rules)],
        "Node Tetap (L2, L3, L5)",
        "#2563eb",
        "circle",
        size=22,
        showlegend=False,
    )
    add_nodes(
        layer_x[4],
        rule_ys,
        [f"f{i + 1}" for i in range(show_rules)],
        [f"Layer 4 Konsekuen Takagi-Sugeno R{i + 1}" for i in range(show_rules)],
        "Node Adaptif (L1 & L4)",
        "#0f766e",
        "square",
        size=22,
        showlegend=False,
    )
    add_nodes(
        layer_x[5],
        out_ys,
        ["Σ"],
        ["Layer 5 Agregasi Output: ŷ = Σ w_bar_k · f_k"],
        "Output Prediksi ISPU",
        "#dc2626",
        "circle",
        size=32,
    )

    # Label header lapisan
    for lx, title in zip(layer_x, layer_titles):
        fig.add_annotation(
            x=lx,
            y=3.45,
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=10.5, color="#0f172a"),
            align="center",
        )

    if truncated:
        fig.add_annotation(
            x=4.5,
            y=-3.25,
            text=f"<i>Menampilkan {show_rules} dari total {n_rules} aturan fuzzy ({partition_type.title()} Partitioning)</i>",
            showarrow=False,
            font=dict(size=10.5, color="#64748b"),
        )

    fig.update_layout(
        title=dict(
            text=f"Arsitektur 5-Lapisan ANFIS ({n_inputs} Input, {n_rules} Aturan Fuzzy)",
            font=dict(size=15, color="#0f172a"),
            x=0.01,
        ),
        template="plotly_white",
        height=440,
        margin=dict(l=16, r=16, t=54, b=24),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#f8fafc",
        xaxis=dict(visible=False, range=[-0.7, 9.7]),
        yaxis=dict(visible=False, range=[-3.6, 4.1]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.04, xanchor="center", x=0.5),
    )
    return fig


def plot_membership_functions(
    model: ANFIS,
    feature_idx: int,
    scaler: MinMaxScalerCustom | None = None,
    show_initial: bool = True,
) -> go.Figure:
    """Menampilkan kurva fungsi keanggotaan (MF) sebelum dan sesudah pelatihan ANFIS."""
    x_grid = np.linspace(0.0, 1.0, 220)
    feature_name = model.feature_names[feature_idx]
    spec = POLLUTANT_SPECS.get(feature_name.lower())
    disp_name = f"{spec['name']} ({spec['unit']})" if spec else feature_name

    if scaler is not None and scaler.data_min_ is not None and scaler.data_range_ is not None:
        x_phys = x_grid * scaler.data_range_[feature_idx] + scaler.data_min_[feature_idx]
        x_axis_vals = x_phys
        x_title = f"Konsentrasi {disp_name}"
    else:
        x_axis_vals = x_grid
        x_title = f"{feature_name} (Ternormalisasi [0, 1])"

    fig = go.Figure()
    trained_curves = model.evaluate_mf_curve(feature_idx, x_grid, use_initial=False)
    init_curves = model.evaluate_mf_curve(feature_idx, x_grid, use_initial=True) if show_initial else []

    max_curves = min(len(trained_curves), 8)
    for idx in range(max_curves):
        label, y_trained = trained_curves[idx]
        color = _PALETTE[idx % len(_PALETTE)]

        if show_initial and idx < len(init_curves):
            _, y_init = init_curves[idx]
            fig.add_trace(
                go.Scatter(
                    x=x_axis_vals,
                    y=y_init,
                    mode="lines",
                    name=f"{label} (Awal)",
                    line=dict(color=color, width=1.6, dash="dash"),
                    opacity=0.55,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x_axis_vals,
                y=y_trained,
                mode="lines",
                name=f"{label} (Terlatih)",
                line=dict(color=color, width=2.8),
            )
        )

    _apply_base_layout(fig, f"Kurva Fungsi Keanggotaan ({model.mf_type.title()}) — {feature_name}", height=390)
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Derajat Keanggotaan μ(x)", range=[-0.04, 1.08])
    return fig


def plot_training_curves(history_df: pd.DataFrame) -> go.Figure:
    """Memplot kurva konvergensi RMSE Latih/Uji serta jadwal step-size adaptif (κ)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Konvergensi Error (RMSE per Epoch)", "Jadwal Step-Size Adaptif Jang (κ)"),
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Scatter(
            x=history_df["epoch"],
            y=history_df["train_rmse"],
            mode="lines+markers",
            name="RMSE Latih",
            line=dict(color="#0f766e", width=2.5),
            marker=dict(size=5),
        ),
        row=1,
        col=1,
    )
    if "val_rmse" in history_df.columns and not history_df["val_rmse"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=history_df["epoch"],
                y=history_df["val_rmse"],
                mode="lines+markers",
                name="RMSE Validasi/Uji",
                line=dict(color="#2563eb", width=2.5, dash="dot"),
                marker=dict(size=5),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=history_df["epoch"],
            y=history_df["kappa"],
            mode="lines",
            name="Step Size (κ)",
            line=dict(color="#d97706", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(217, 119, 6, 0.12)",
        ),
        row=1,
        col=2,
    )

    _apply_base_layout(fig, "Dinamika Pelatihan Hybrid Learning (Forward LSE + Backward GD)", height=370)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="RMSE (Skor ISPU)", row=1, col=1)
    fig.update_yaxes(title_text="Learning Rate κ", row=1, col=2)
    return fig


def plot_actual_vs_predicted_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Scatter plot ISPU Aktual vs Prediksi ANFIS berwarna sesuai kategori Permen LHK."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    cats = [classify_ispu(v)["name"] for v in yt]

    fig = go.Figure()
    for cat in ISPU_CATEGORIES:
        mask = np.array([c == cat["name"] for c in cats])
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=yt[mask],
                y=yp[mask],
                mode="markers",
                name=cat["name"],
                marker=dict(color=cat["color"], size=7, opacity=0.82, line=dict(width=0.6, color="#ffffff")),
            )
        )

    lo = float(min(np.min(yt), np.min(yp)))
    hi = float(max(np.max(yt), np.max(yp)))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Garis Ideal (y = x)",
            line=dict(color="#0f172a", width=2, dash="dash"),
        )
    )

    _apply_base_layout(fig, "ISPU Aktual vs Prediksi ANFIS (Data Uji)", height=410)
    fig.update_xaxes(title_text="ISPU Aktual")
    fig.update_yaxes(title_text="ISPU Prediksi ANFIS")
    return fig


def plot_actual_vs_predicted_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = 80,
) -> go.Figure:
    """Grafik garis perbandingan ISPU Aktual dan Prediksi ANFIS pada sampel data uji."""
    n = min(len(y_true), max_points)
    idx = np.arange(1, n + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=y_true[:n],
            mode="lines+markers",
            name="ISPU Aktual",
            line=dict(color="#2563eb", width=2.2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=y_pred[:n],
            mode="lines+markers",
            name="Prediksi ANFIS",
            line=dict(color="#dc2626", width=2.0, dash="dot"),
            marker=dict(size=4),
        )
    )

    _apply_base_layout(fig, f"Perbandingan Nilai Aktual vs Prediksi ({n} Sampel Uji Pertama)", height=380)
    fig.update_xaxes(title_text="Indeks Sampel Uji")
    fig.update_yaxes(title_text="Skor ISPU")
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Plot analisis residual (histogram error & sebaran residual terhadap prediksi)."""
    residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distribusi Error (Aktual - Prediksi)", "Residual vs Prediksi"),
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=28, name="Residual", marker_color="#0f766e", opacity=0.85),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="Sebaran Error",
            marker=dict(color="#2563eb", size=6, opacity=0.65),
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#dc2626", row=1, col=2)

    _apply_base_layout(fig, "Diagnostik Residual Prediksi ANFIS", height=360)
    fig.update_xaxes(title_text="Residual (poin ISPU)", row=1, col=1)
    fig.update_yaxes(title_text="Frekuensi", row=1, col=1)
    fig.update_xaxes(title_text="Prediksi ISPU", row=1, col=2)
    fig.update_yaxes(title_text="Residual (poin ISPU)", row=1, col=2)
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> go.Figure:
    """Heatmap Confusion Matrix 5 kategori ISPU resmi Permen LHK No. 14/2020."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=12),
            hoverongaps=False,
        )
    )
    _apply_base_layout(fig, "Confusion Matrix Kategori ISPU (Data Uji)", height=410)
    fig.update_xaxes(title_text="Kategori Prediksi ANFIS", side="bottom")
    fig.update_yaxes(title_text="Kategori Aktual", autorange="reversed")
    fig.update_layout(margin=dict(l=110, r=24, t=54, b=64))
    return fig


def plot_rule_contributions(rule_df: pd.DataFrame, top_k: int = 10) -> go.Figure:
    """Menampilkan Top-K aturan fuzzy yang paling aktif beserta kontribusi outputnya."""
    top_df = rule_df.head(top_k).iloc[::-1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Bobot Aktif Ternormalisasi (w_bar)", "Kontribusi Output (w_bar · f_k)"),
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Bar(
            y=top_df["Aturan"],
            x=top_df["w_bar (Layer 3)"],
            orientation="h",
            name="w_bar",
            marker_color="#0f766e",
            hovertext=top_df["Antecedent"],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=top_df["Aturan"],
            x=top_df["w_bar * f_k"],
            orientation="h",
            name="w_bar · f_k",
            marker_color="#2563eb",
            hovertext=top_df["Antecedent"],
        ),
        row=1,
        col=2,
    )

    _apply_base_layout(fig, f"Top-{len(top_df)} Aturan Fuzzy Paling Berpengaruh", height=370)
    fig.update_xaxes(title_text="w_bar [0–1]", row=1, col=1)
    fig.update_xaxes(title_text="Kontribusi Skor ISPU", row=1, col=2)
    return fig


def plot_baseline_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Membandingkan RMSE dan Akurasi Kategori antara ANFIS dan model baseline."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Perbandingan RMSE Uji (Lebih Rendah Lebih Baik)", "Akurasi Kategori ISPU (%)"),
        horizontal_spacing=0.10,
    )

    colors = ["#0f766e" if "ANFIS" in m else "#64748b" for m in comparison_df["Model"]]

    fig.add_trace(
        go.Bar(
            x=comparison_df["Model"],
            y=comparison_df["RMSE"],
            text=comparison_df["RMSE"].round(2),
            textposition="outside",
            marker_color=colors,
            name="RMSE",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=comparison_df["Model"],
            y=comparison_df["Akurasi_Kategori"],
            text=comparison_df["Akurasi_Kategori"].round(1).astype(str) + "%",
            textposition="outside",
            marker_color=colors,
            name="Akurasi (%)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    _apply_base_layout(fig, "Komparasi Kinerja ANFIS vs Metode Pembanding", height=390)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="Akurasi (%)", range=[0, 108], row=1, col=2)
    return fig


def plot_data_exploration_categories(df: pd.DataFrame, target_col: str = "ispu") -> go.Figure:
    """Distribusi jumlah observasi per kategori ISPU resmi Permen LHK No. 14/2020."""
    cats = [classify_ispu(v)["name"] for v in df[target_col].to_numpy()]
    counts = pd.Series(cats).value_counts()

    ordered_names = [c["name"] for c in ISPU_CATEGORIES]
    ordered_colors = [c["color"] for c in ISPU_CATEGORIES]
    ordered_vals = [int(counts.get(name, 0)) for name in ordered_names]

    fig = go.Figure(
        go.Bar(
            x=ordered_names,
            y=ordered_vals,
            marker_color=ordered_colors,
            text=ordered_vals,
            textposition="outside",
        )
    )
    _apply_base_layout(fig, "Distribusi Kategori Kualitas Udara (ISPU)", height=360)
    fig.update_xaxes(title_text="Kategori ISPU")
    fig.update_yaxes(title_text="Jumlah Observasi", range=[0, max(ordered_vals or [10]) * 1.16])
    return fig


def plot_data_exploration_corr(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    """Heatmap matriks korelasi Pearson antar variabel polutan dan target ISPU."""
    corr = df[cols].corr().round(2)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.to_numpy(),
            x=cols,
            y=cols,
            colorscale="Tealrose",
            zmin=-1.0,
            zmax=1.0,
            text=corr.to_numpy(),
            texttemplate="%{text:.2f}",
            textfont=dict(size=11),
        )
    )
    _apply_base_layout(fig, "Korelasi Antar Polutan Udara & Target ISPU", height=380)
    fig.update_yaxes(autorange="reversed")
    return fig
