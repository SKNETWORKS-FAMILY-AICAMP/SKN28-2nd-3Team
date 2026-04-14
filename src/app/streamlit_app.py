# streamlit_app.py

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import streamlit as st

# ------------------------------------------------------
# 프로젝트 루트 경로를 sys.path에 추가
# - 로컬 / GitHub / Streamlit Cloud 환경에서 import 안정성 확보
# ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------
# 섹션 import
# ------------------------------------------------------
from src.app.sections.eda_section import render as render_eda
from src.app.sections.model_section import render as render_model
from src.app.sections.overview_section import render as render_overview
from src.app.sections.prediction_section import render as render_prediction
from src.app.sections.xai_section import render as render_xai

# 한글 폰트 설정 함수가 있다면 사용
try:
    from src.utils.plot_utils import set_korean_font
except ImportError:
    set_korean_font = None


# ------------------------------------------------------
# 페이지 기본 설정
# ------------------------------------------------------
st.set_page_config(
    page_title="설명 가능한 AI 기반 SaaS 고객 이탈 예측 및 고객 유지 전략 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_global_settings() -> None:
    """앱 시작 시 전역 설정 적용"""
    if set_korean_font is not None:
        try:
            set_korean_font()
        except Exception:
            # 폰트 설정 실패 시 앱 전체가 죽지 않도록 방어
            pass


def inject_custom_css() -> None:
    """앱 전역 CSS"""
    st.markdown(
        dedent(
            """
            <style>
            /* --------------------------------------------------
               전체 레이아웃
            -------------------------------------------------- */
            .block-container {
                padding-top: 2.8rem;
                padding-bottom: 2.2rem;
                max-width: 1240px;
            }

            /* --------------------------------------------------
               사이드바
            -------------------------------------------------- */
            [data-testid="stSidebar"] {
                min-width: 300px;
                max-width: 300px;
                border-right: 1px solid #eef2f7;
            }

            [data-testid="stSidebarCollapsedControl"] {
                display: none;
            }

            /* --------------------------------------------------
               기본 타이포
            -------------------------------------------------- */
            html, body, [class*="css"] {
                word-break: keep-all;
            }

            h1, h2, h3, h4 {
                color: #111827;
                letter-spacing: -0.01em;
            }

            /* --------------------------------------------------
               Hero
            -------------------------------------------------- */
            .hero-box {
                background: linear-gradient(135deg, #0f172a 0%, #23344d 52%, #334155 100%);
                color: #ffffff;
                padding: 2.1rem 2rem 1.7rem 2rem;
                border-radius: 24px;
                margin-top: 0.2rem;
                margin-bottom: 1.6rem;
                box-shadow: 0 14px 36px rgba(15, 23, 42, 0.16);
            }

            .hero-eyebrow {
                font-size: 0.95rem;
                opacity: 0.86;
                margin-bottom: 0.45rem;
                line-height: 1.5;
            }

            .hero-title {
                font-size: 2.15rem;
                font-weight: 800;
                line-height: 1.3;
                margin-bottom: 0.7rem;
                letter-spacing: -0.02em;
            }

            .hero-desc {
                font-size: 1.02rem;
                opacity: 0.94;
                line-height: 1.8;
                margin: 0;
            }

            /* --------------------------------------------------
               공통 카드
            -------------------------------------------------- */
            .section-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 1.15rem 1.2rem;
                margin-bottom: 1rem;
                box-sizing: border-box;
                overflow: hidden;
            }

            .mini-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 1rem 1rem 0.85rem 1rem;
                box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
                box-sizing: border-box;
                margin-bottom: 1rem;
                overflow: hidden;
            }

            .equal-card {
                min-height: 220px;
            }

            .mini-label {
                color: #64748b;
                font-size: 0.9rem;
                font-weight: 600;
                margin-bottom: 0.25rem;
                line-height: 1.45;
            }

            .mini-value {
                font-size: 2rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.15;
                margin-bottom: 0.2rem;
                letter-spacing: -0.02em;
            }

            .mini-sub {
                color: #64748b;
                font-size: 0.9rem;
                line-height: 1.45;
            }

            .feature-card-title {
                font-size: 1.15rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.35;
                margin-top: 0.55rem;
                word-break: break-word;
                overflow-wrap: anywhere;
            }

            /* --------------------------------------------------
               사이드바 텍스트
            -------------------------------------------------- */
            .sidebar-title {
                font-size: 1.05rem;
                font-weight: 800;
                color: #111827;
                margin-bottom: 0.35rem;
            }

            .sidebar-sub {
                color: #64748b;
                font-size: 0.89rem;
                line-height: 1.55;
                margin-bottom: 1rem;
            }

            .flow-box {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 0.95rem 1rem;
                margin-top: 1rem;
                line-height: 1.85;
                color: #334155;
                font-size: 0.92rem;
            }

            /* --------------------------------------------------
               Streamlit 기본 요소 보정
            -------------------------------------------------- */
            .stRadio > div {
                gap: 0.65rem;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.4rem;
            }

            .stTabs [data-baseweb="tab"] {
                padding-left: 0.65rem;
                padding-right: 0.65rem;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
            }

            /* --------------------------------------------------
               expander / info / metric 느낌 정리
            -------------------------------------------------- */
            div[data-testid="stExpander"] {
                border-radius: 14px;
                border: 1px solid #e5e7eb;
                overflow: hidden;
            }

            div[data-testid="stInfo"] {
                border-radius: 14px;
            }

            /* --------------------------------------------------
               반응형
            -------------------------------------------------- */
            @media (max-width: 1100px) {
                .block-container {
                    max-width: 100%;
                    padding-top: 2.2rem;
                }

                .hero-title {
                    font-size: 1.9rem;
                }
            }

            @media (max-width: 900px) {
                [data-testid="stSidebar"] {
                    min-width: 260px;
                    max-width: 260px;
                }

                .hero-box {
                    padding: 1.7rem 1.4rem 1.45rem 1.4rem;
                }

                .hero-title {
                    font-size: 1.65rem;
                }

                .hero-desc {
                    font-size: 0.98rem;
                }

                .equal-card {
                    min-height: unset;
                }
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """상단 히어로 영역"""
    st.markdown(
        dedent(
            """
            <div class="hero-box">
                <div class="hero-eyebrow">
                    SKN28-2nd-3Team · 머신러닝/딥러닝 프로젝트 · RavenStack Synthetic SaaS Dataset
                </div>
                <div class="hero-title">
                    설명 가능한 AI 기반 SaaS 고객 이탈 예측 및 고객 유지 전략 분석
                </div>
                <p class="hero-desc">
                    단순히 churn을 맞히는 데서 멈추지 않고,
                    어떤 고객이 왜 이탈 위험이 높은지를 해석하고
                    그 결과를 바탕으로 실행 가능한 고객 유지 전략까지 연결하는 것을 목표로 한 대시보드이다.
                </p>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """사이드바 네비게이션"""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">Dashboard Navigation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-sub">프로젝트 흐름에 따라 핵심 결과를 순서대로 확인할 수 있도록 구성했다.</div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "페이지 이동",
            [
                "프로젝트 개요",
                "EDA",
                "모델 성능",
                "이탈 요인 & 유지 전략",
                "고객별 예측",
            ],
            label_visibility="collapsed",
        )

        st.markdown(
            dedent(
                """
                <div class="flow-box">
                    <b>분석 흐름</b><br>
                    ① 문제 정의<br>
                    ② 이탈 고객 특성 파악<br>
                    ③ 예측 모델 비교 및 threshold 조정<br>
                    ④ XAI 기반 주요 이탈 신호 해석<br>
                    ⑤ 고객 유지 전략 제안
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

    return page


def render_page(page: str) -> None:
    """선택된 페이지 렌더링"""
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
    render_header()
    selected_page = render_sidebar()
    render_page(selected_page)


if __name__ == "__main__":
    main()