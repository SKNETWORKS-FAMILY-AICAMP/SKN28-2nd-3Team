# streamlit_app.py

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.sections.eda_section import render as render_eda
from src.app.sections.model_section import render as render_model
from src.app.sections.overview_section import render as render_overview
from src.app.sections.prediction_section import render as render_prediction
from src.app.sections.xai_section import render as render_xai

try:
    from src.utils.plot_utils import set_korean_font
except ImportError:
    set_korean_font = None

st.set_page_config(
    page_title="SaaS 고객 이탈 예측 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_global_settings() -> None:
    if set_korean_font is not None:
        try:
            set_korean_font()
        except Exception:
            pass


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

        /* ── 전체 ── */
        html, body, [class*="css"] {
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
            word-break: keep-all;
        }

        .block-container {
            padding: 3.2rem 2.5rem 3rem 2.5rem;
            max-width: 1280px;
        }

        /* ── 사이드바 ── */
        [data-testid="stSidebar"] {
            min-width: 290px;
            max-width: 290px;
            background: #0f172a;
            border-right: none;
        }

        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }

        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        [data-testid="stSidebar"] .stRadio label {
            padding: 0.55rem 0.9rem;
            border-radius: 10px;
            transition: background 0.15s;
            font-size: 0.92rem;
            font-weight: 500;
            white-space: nowrap;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255,255,255,0.08) !important;
        }

        /* ── 히어로 ── */
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1e40af 100%);
            border-radius: 20px;
            padding: 2rem 2.4rem 1.8rem;
            margin-top: 0.4rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: '';
            position: absolute;
            top: -60px;
            right: -60px;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(96,165,250,0.15) 0%, transparent 70%);
            border-radius: 50%;
        }

        .hero-badge {
            display: inline-block;
            background: rgba(96,165,250,0.18);
            color: #93c5fd;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(96,165,250,0.25);
            margin-bottom: 0.9rem;
        }

        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.3;
            margin: 0 0 0.75rem;
            letter-spacing: -0.025em;
        }

        .hero-desc {
            font-size: 0.97rem;
            color: rgba(255,255,255,0.72);
            line-height: 1.75;
            margin: 0;
            max-width: 680px;
        }

        /* ── 카드 ── */
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
        }

        .card-gray {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
        }

        .card h4 {
            font-size: 0.92rem;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0 0 0.6rem;
        }

        /* ── stat 카드 ── */
        .stat-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.2rem 1.3rem 1rem;
            margin-bottom: 0.75rem;
        }

        .stat-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }

        .stat-sub {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-top: 0.3rem;
        }

        /* ── 섹션 타이틀 ── */
        .section-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 0.3rem;
            letter-spacing: -0.02em;
        }

        .section-sub {
            font-size: 0.92rem;
            color: #64748b;
            margin: 0 0 1.5rem;
        }

        /* ── 구분선 ── */
        .divider {
            height: 1px;
            background: #e2e8f0;
            margin: 1.5rem 0;
        }

        /* ── 노트 ── */
        .note {
            font-size: 0.87rem;
            color: #64748b;
            line-height: 1.65;
            text-align: center;
            margin: 0.4rem 0 1.2rem;
        }

        /* ── 태그 ── */
        .tag {
            display: inline-block;
            background: #eff6ff;
            color: #2563eb;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            margin-right: 0.3rem;
        }

        .tag-red   { background: #fef2f2; color: #dc2626; }
        .tag-green { background: #f0fdf4; color: #16a34a; }
        .tag-amber { background: #fffbeb; color: #d97706; }

        /* ── Streamlit 기본 요소 ── */
        h1, h2, h3, h4 {
            color: #0f172a;
            letter-spacing: -0.015em;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            border-bottom: 2px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 8px 8px 0 0;
            color: #64748b;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
            border: 1px solid #e2e8f0 !important;
        }

        div[data-testid="stInfo"],
        div[data-testid="stSuccess"],
        div[data-testid="stWarning"] {
            border-radius: 12px;
            font-size: 0.92rem;
        }

        .stMetric {
            background: #f8fafc;
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }

        /* ── 반응형 ── */
        @media (max-width: 900px) {
            .block-container {
                padding: 2.4rem 1.2rem 2.4rem 1.2rem;
            }

            .hero-title {
                font-size: 1.55rem;
            }

            [data-testid="stSidebar"] {
                min-width: 240px;
                max-width: 240px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">SKN28 · 2nd Project · Team 3</div>
            <div class="hero-title">SaaS 고객 이탈 예측 &amp; 유지 전략 대시보드</div>
            <p class="hero-desc">
                어떤 고객이 왜 이탈 위험이 높은지 해석하고,
                그 결과를 실행 가능한 고객 유지 전략으로 연결합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 1.2rem 0.8rem 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 1rem;">
                <div style="font-size:1.05rem; font-weight:800; color:#f1f5f9;">📊 Navigation</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.3rem;">분석 흐름 순서로 구성</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "페이지",
            ["프로젝트 개요", "EDA", "모델 성능", "이탈 요인 & 유지 전략", "고객별 예측"],
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div style="margin-top:2rem; padding: 1rem; background:rgba(255,255,255,0.05);
                        border-radius:12px; font-size:0.82rem; color:#94a3b8; line-height:2;">
                <div style="color:#cbd5e1; font-weight:700; margin-bottom:0.4rem;">분석 흐름</div>
                ① 문제 정의 &amp; 데이터<br>
                ② 이탈 고객 특성 파악<br>
                ③ 모델 비교 &amp; threshold<br>
                ④ XAI 이탈 신호 해석<br>
                ⑤ 고객 유지 전략 제안
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


def render_page(page: str) -> None:
    if page == "프로젝트 개요":
        render_overview()
    elif page == "EDA":
        render_eda()
    elif page == "모델 성능":
        render_model()
    elif page == "이탈 요인 & 유지 전략":
        render_xai()
    elif page == "고객별 예측":
        render_prediction()
    else:
        st.error("알 수 없는 페이지입니다.")


def main() -> None:
    apply_global_settings()
    inject_custom_css()
    selected_page = render_sidebar()
    render_header()
    render_page(selected_page)


if __name__ == "__main__":
    main()