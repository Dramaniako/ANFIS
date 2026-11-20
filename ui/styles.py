"""Konfigurasi gaya visual CSS dan komponen badge status ISPU untuk Streamlit."""

from __future__ import annotations

import html
import streamlit as st

from src.ispu_standard import classify_ispu


def inject_custom_css() -> None:
    """Menyuntikkan CSS kustom agar tampilan GUI rapi, konsisten, dan informatif."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #0f766e 100%);
            color: #ffffff;
            padding: 1.65rem 1.9rem;
            border-radius: 14px;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        }
        .hero-banner h1 {
            margin: 0 0 0.45rem 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: #ffffff;
        }
        .hero-banner p {
            margin: 0;
            font-size: 0.98rem;
            color: #e2e8f0;
            line-height: 1.55;
        }
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #0f766e;
            border-radius: 10px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
            height: 100%;
        }
        .kpi-label {
            font-size: 0.80rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }
        .kpi-sub {
            font-size: 0.82rem;
            color: #475569;
        }
        .step-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem 1.1rem;
            height: 100%;
        }
        .step-num {
            display: inline-block;
            background: #0f766e;
            color: #ffffff;
            font-weight: 700;
            font-size: 0.78rem;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            margin-bottom: 0.45rem;
        }
        .step-title {
            font-weight: 700;
            font-size: 0.98rem;
            color: #0f172a;
            margin-bottom: 0.3rem;
        }
        .step-desc {
            font-size: 0.86rem;
            color: #475569;
            line-height: 1.45;
        }
        .ispu-badge-box {
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            margin-bottom: 0.8rem;
        }
        .ispu-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }
        .sidebar-status {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.8rem 0.9rem;
            font-size: 0.83rem;
            color: #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtitle: str = "", accent: str = "#0f766e") -> None:
    """Merender kartu metrik KPI dengan border aksen yang rapi."""
    safe_label = html.escape(label)
    safe_val = html.escape(value)
    safe_sub = html.escape(subtitle)
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color: {accent};">
            <div class="kpi-label">{safe_label}</div>
            <div class="kpi-value">{safe_val}</div>
            <div class="kpi-sub">{safe_sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ispu_badge(score: float, title: str = "Prediksi Skor ISPU") -> None:
    """Merender panel skor ISPU lengkap dengan warna kategori resmi Permen LHK 14/2020 dan rekomendasi."""
    cat = classify_ispu(score)
    safe_title = html.escape(title)
    safe_cat = html.escape(cat["name"])
    safe_rec = html.escape(cat["recommendation"])
    st.markdown(
        f"""
        <div class="ispu-badge-box" style="background-color: {cat['bg_light']}; border-left: 6px solid {cat['color']};">
            <div style="font-size: 0.82rem; font-weight: 600; color: #334155; margin-bottom: 0.25rem;">
                {safe_title}
            </div>
            <div style="display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 0.5rem;">
                <span style="font-size: 2.0rem; font-weight: 800; color: #0f172a;">{score:.2f}</span>
                <span class="ispu-pill" style="background-color: {cat['color']}; color: {cat['text_color']};">
                    {safe_cat} ({cat['badge_label']} • {cat['range_label']})
                </span>
            </div>
            <div style="font-size: 0.86rem; color: #1e293b; line-height: 1.45;">
                <b>Rekomendasi Kesehatan:</b> {safe_rec}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
