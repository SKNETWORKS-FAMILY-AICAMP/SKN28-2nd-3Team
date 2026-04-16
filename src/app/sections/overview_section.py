# overview_section.py

from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

from textwrap import dedent  # 여러 줄 문자열 들여쓰기 정리를 위한 유틸 (현재는 사용되지 않지만 확장 대비)

import streamlit as st  # Streamlit UI 구성 라이브러리

# 대시보드에 필요한 데이터 로드 함수들
from src.app.utils.load_data import (
    load_model_comparison_tuned,  # 튜닝된 모델 성능 데이터 로드
    load_train_table,             # 학습용 데이터 테이블 로드
    load_xai_summary,             # XAI 요약 결과 데이터 로드
)


# KPI 스타일 카드(stat-card)를 출력하는 함수
def _stat(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>   <!-- KPI 이름 -->
            <div class="stat-value">{value}</div>   <!-- KPI 값 -->
            {"" if not sub else f'<div class="stat-sub">{sub}</div>'}  <!-- 보조 설명 (옵션) -->
        </div>
        """,
        unsafe_allow_html=True,  # HTML 스타일 적용 허용
    )


# 카드 형태 UI를 출력하는 함수 (gray 여부에 따라 스타일 변경)
def _card(title: str, body: str, gray: bool = False) -> None:
    cls = "card-gray" if gray else "card"  # 카드 스타일 클래스 선택
    st.markdown(
        f"""
        <div class="{cls}">
            <h4>{title}</h4>   <!-- 카드 제목 -->
            <div style="color:#334155; font-size:0.93rem; line-height:1.75;">{body}</div>  <!-- 본문 -->
        </div>
        """,
        unsafe_allow_html=True,
    )


# Streamlit에서 호출되는 메인 렌더링 함수
def render() -> None:
    # 데이터 로드
    df = load_train_table()                      # 학습 데이터
    tuned_df = load_model_comparison_tuned()     # 튜닝된 모델 결과
    xai_df = load_xai_summary()                  # XAI 요약 결과

    # churn 비율 계산 (데이터 없을 경우 0)
    churn_rate = df["churn_flag"].mean() * 100 if not df.empty else 0.0

    # 초기값 설정
    best_model, best_f1, best_threshold = "-", "-", "-"

    # 튜닝 결과가 존재하면 최고 성능 모델 선택
    if not tuned_df.empty:
        row = tuned_df.sort_values("f1", ascending=False).iloc[0]  # F1 기준 최고 행 선택
        best_model = str(row["model"])                             # 모델명
        best_f1 = f"{row['f1']:.3f}"                               # F1 점수
        best_threshold = f"{row['threshold']:.2f}"                 # threshold 값

    # XAI에서 가장 중요한 feature 추출
    top_signal = str(xai_df.iloc[0]["feature"]) if not xai_df.empty and "feature" in xai_df.columns else "-"

    # ── 타이틀 영역 ──
    st.markdown(
        """
        <div class="section-title">프로젝트 개요</div>
        <div class="section-sub">RavenStack Synthetic SaaS Dataset 기반 고객 이탈 예측</div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI 카드 4개 ──
    # 주요 지표를 한눈에 보기 위한 영역
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        _stat("분석 고객 수", f"{len(df):,}명", "account 단위")  # 고객 수

    with c2:
        _stat("입력 피처 수", f"{df.shape[1]:,}개", "전처리 후 기준")  # feature 수

    with c3:
        _stat("Churn 비율", f"{churn_rate:.1f}%", "불균형 분류 문제")  # 이탈 비율

    with c4:
        _stat("최적 Threshold", best_threshold, f"{best_model} · F1 {best_f1}")  # 최적 threshold + 모델

    # 구분선
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 문제 정의 + 핵심 메시지 ──
    col1, col2 = st.columns(2)

    # 왼쪽: 문제 정의
    with col1:
        _card(
            "문제 정의",
            f"""
            SaaS 환경에서는 이탈을 사후 확인하는 것보다
            <strong>이탈 가능성이 높은 고객을 미리 탐지해 선제 개입</strong>하는 것이 중요합니다.<br><br>
            본 프로젝트는 account 단위 데이터를 기반으로 churn을 예측하고,
            SHAP 해석과 연결해 <strong>실질적인 retention 전략</strong>까지 제시합니다.
            """,
        )

    # 오른쪽: 핵심 결과 요약
    with col2:
        _card(
            "핵심 결과 요약",
            f"""
            <span class="tag">최적 모델</span> {best_model}<br>
            <span class="tag">운영 Threshold</span> {best_threshold}<br>
            <span class="tag">대표 이탈 신호</span> {top_signal}<br><br>
            <strong>예측 → 해석 → 유지 전략 연결</strong>이 이 대시보드의 목표입니다.
            """,
            gray=True,  # 회색 카드 스타일 적용
        )

    # ── 해석 + 분석 흐름 ──
    col3, col4 = st.columns(2)

    # 왼쪽: 모델 운영 기준 설명
    with col3:
        _card(
            "모델 운영 기준",
            f"""
            단순 accuracy가 아닌 <strong>이탈 고객을 얼마나 놓치지 않는지(recall)</strong>와
            실무 적용 가능한 threshold 설정이 핵심입니다.<br><br>
            기본값 0.5 대신 <strong>F1 기준 threshold tuning 결과</strong>를 별도 비교했습니다.
            """,
        )

    # 오른쪽: 전체 분석 흐름
    with col4:
        _card(
            "분석 흐름",
            """
            <ol style="margin:0; padding-left:1.1rem; line-height:2;">
                <li>EDA — churn 고객 특성 확인</li>
                <li>ML/DL 모델 비교 &amp; threshold 설정</li>
                <li>XAI — 주요 이탈 요인 해석</li>
                <li>고객 유지 전략으로 연결</li>
            </ol>
            """,
            gray=True,
        )

    # ── 핵심 질문 3개 ──
    # 이 대시보드가 해결하려는 질문을 명확히 제시
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.85rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.8rem;">이 대시보드가 답하는 질문</div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)

    q1.info("어떤 고객이 이탈 위험이 높은가?")
    q2.info("모델은 무엇을 근거로 판단했는가?")
    q3.info("그 고객을 유지하기 위해 무엇을 할 수 있는가?")