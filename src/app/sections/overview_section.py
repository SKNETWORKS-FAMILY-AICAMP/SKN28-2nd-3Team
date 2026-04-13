from __future__ import annotations

from textwrap import dedent

import streamlit as st

from src.app.utils.load_data import (
    load_model_comparison_tuned,
    load_train_table,
    load_xai_summary,
)


def _mini_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        dedent(f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
            <div class="mini-sub">{sub}</div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def _card(title: str, body_html: str) -> None:
    body_html = dedent(body_html).strip()

    st.markdown(
        dedent(f"""
        <div class="section-card equal-card">
            <h4 style="margin-top:0; margin-bottom:0.9rem;">{title}</h4>
            {body_html}
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render() -> None:
    df = load_train_table()
    tuned_df = load_model_comparison_tuned()
    xai_df = load_xai_summary()

    churn_rate = df["churn_flag"].mean() * 100 if not df.empty else 0.0

    best_model = "-"
    best_f1 = "-"
    best_threshold = "-"

    if not tuned_df.empty:
        best_row = tuned_df.sort_values("f1", ascending=False).iloc[0]
        best_model = str(best_row["model"])
        best_f1 = f"{best_row['f1']:.3f}"
        best_threshold = f"{best_row['threshold']:.2f}"

    top_signal = "-"
    if not xai_df.empty and "feature" in xai_df.columns:
        top_signal = str(xai_df.iloc[0]["feature"])

    st.markdown("## 프로젝트 개요")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _mini_card("고객 수", f"{len(df):,}", "account 기준 분석")
    with c2:
        _mini_card("입력 변수 수", f"{df.shape[1]:,}", "전처리 후 학습 테이블 기준")
    with c3:
        _mini_card("Churn 비율", f"{churn_rate:.1f}%", "불균형 분류 문제")
    with c4:
        _mini_card("최적 운영 기준", best_threshold, f"{best_model} · F1 {best_f1}")

    st.markdown("")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        _card(
            "문제 정의",
            """
<p style="line-height:1.8; margin-bottom:0;">
SaaS 환경에서는 고객 이탈을 사후적으로 확인하는 것보다,
<b>이탈 가능성이 높은 고객을 미리 탐지하고 선제적으로 개입</b>하는 것이 더 중요하다.
본 프로젝트는 account 단위 데이터를 기반으로 churn을 예측하고,
예측 결과를 SHAP 기반 설명과 연결하여 <b>실질적인 retention 전략</b>까지 제시하는 것을 목표로 한다.
</p>
            """,
        )
    with row1_col2:
        _card(
            "한눈에 보는 프로젝트 메시지",
            f"""
<ul style="line-height:1.9; margin-bottom:0; padding-left:1.2rem;">
    <li>최적 모델 후보: <b>{best_model}</b></li>
    <li>운영 기준 threshold: <b>{best_threshold}</b></li>
    <li>대표 이탈 신호: <b>{top_signal}</b></li>
    <li>최종 목표: <b>예측 + 해석 + 유지 전략 연결</b></li>
</ul>
            """,
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        _card(
            "핵심 해석",
            """
<p style="line-height:1.8; margin-bottom:0;">
이번 프로젝트에서 중요한 것은 단순 accuracy가 아니라,
<b>이탈 고객을 얼마나 놓치지 않는지(recall)</b>와
<b>실무에서 사용할 수 있는 threshold를 어떻게 잡을지</b>이다.
따라서 우리는 기본 threshold 0.5 결과와 함께
<b>F1 기준 threshold tuning 결과</b>를 별도로 비교하였다.
</p>
            """,
        )
    with row2_col2:
        _card(
            "전체 흐름",
            """
<ol style="line-height:1.9; margin-bottom:0; padding-left:1.2rem;">
    <li>EDA로 churn 고객 특성 확인</li>
    <li>ML/DL 성능 비교 및 threshold 재설정</li>
    <li>XAI로 주요 이탈 요인 해석</li>
    <li>고객 유지 전략으로 연결</li>
</ol>
            """,
        )

    st.markdown("### 이번 대시보드에서 답하고 싶은 질문")
    q1, q2, q3 = st.columns(3)
    q1.info("어떤 고객이 이탈 위험이 높은가?")
    q2.info("모델은 무엇을 근거로 그렇게 판단했는가?")
    q3.info("그 고객을 유지하기 위해 무엇을 할 수 있는가?")