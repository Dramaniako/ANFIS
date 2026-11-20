"""Aplikasi Utama AirFuzzy-ANFIS: Prediksi Polusi Udara ISPU Indonesia.

Jalankan melalui terminal:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from ui.page_data import render_page_data
from ui.page_evaluation import render_page_evaluation
from ui.page_home import render_page_home
from ui.page_info import render_page_info
from ui.page_results import render_page_results
from ui.page_training import render_page_training
from ui.state import initialize_session_state
from ui.styles import inject_custom_css

st.set_page_config(
    page_title="AirFuzzy-ANFIS | Prediksi Polusi Udara ISPU",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "1. Beranda": render_page_home,
    "2. Input & Eksplorasi Data": render_page_data,
    "3. Pelatihan ANFIS 5-Layer": render_page_training,
    "4. Hasil & Trace Layer 1–5": render_page_results,
    "5. Evaluasi & Komparasi": render_page_evaluation,
    "6. Informasi Sistem & Teori": render_page_info,
}


def main() -> None:
    inject_custom_css()
    initialize_session_state()

    with st.sidebar:
        st.markdown("## AirFuzzy-ANFIS")
        st.caption("Teknologi Logika Fuzzy • Studi Kasus Polusi Udara ISPU")

        selected_page = st.radio(
            "Navigasi Halaman",
            options=list(PAGES.keys()),
            key="active_page",
        )

        st.markdown("---")
        model = st.session_state["anfis_model"]
        summary = model.summary()
        test_metrics = st.session_state["test_metrics"]
        test_cls = st.session_state["test_cls_metrics"]

        st.markdown(
            f"""
            <div class="sidebar-status">
                <b>Status Model Aktif</b><br/>
                Partisi: <code>{summary['partition_type']}</code> ({summary['mf_type']})<br/>
                Aturan Fuzzy: <b>{summary['n_rules']}</b> ({summary['n_inputs']} input)<br/>
                RMSE Uji: <b>{test_metrics['RMSE']:.2f}</b> | R²: <b>{test_metrics['R2']:.4f}</b><br/>
                Akurasi Kategori: <b>{test_cls['accuracy'] * 100.0:.1f}%</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    PAGES[selected_page]()


if __name__ == "__main__":
    main()
