# eda_section.py

from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 함

import pandas as pd  # 데이터프레임 처리용 라이브러리
import streamlit as st  # Streamlit UI 구성 라이브러리

# 모델 성능 관련 CSV 데이터를 불러오는 함수들
from src.app.utils.load_data import (
    load_model_comparison,              # 기본 모델 성능 비교 데이터
    load_model_comparison_tuned,        # threshold 튜닝된 모델 성능 데이터
    load_threshold_metrics_all_models,  # threshold별 성능 변화 데이터
)


# 안내 문구를 카드 스타일(note 클래스)로 출력하는 함수
def _note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# 특정 metric 기준으로 가장 성능이 좋은 행(Top 1)을 반환하는 함수
def _safe_top(df: pd.DataFrame, metric: str) -> pd.Series | None:
    # 데이터가 없거나 metric 컬럼이 없으면 None 반환
    if df.empty or metric not in df.columns:
        return None
    # metric 기준으로 내림차순 정렬 후 가장 위 행 반환
    return df.sort_values(metric, ascending=False).iloc[0]


# Streamlit에서 실제로 호출되는 메인 렌더링 함수
def render() -> None:
    # CSV 데이터 로드
    base_df = load_model_comparison()          # 기본 성능 비교 데이터
    tuned_df = load_model_comparison_tuned()   # 튜닝된 성능 비교 데이터
    threshold_df = load_threshold_metrics_all_models()  # threshold별 성능 데이터

    # 기본 모델 중 ROC-AUC 기준 최고 모델 선택
    top_base = _safe_top(base_df, "roc_auc")

    # 튜닝된 모델 중 F1 기준 최고 모델 선택
    top_tuned = _safe_top(tuned_df, "f1")

    # 페이지 상단 제목 및 설명 출력
    st.markdown(
        """
        <div class="section-title">모델 성능</div>
        <div class="section-sub">어떤 모델이 좋은지보다, 어떤 기준으로 운영할지를 결정하는 단계입니다</div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI 3개 ──
    # 3개의 컬럼으로 주요 성능 지표를 시각적으로 표시
    c1, c2, c3 = st.columns(3)

    # 1번 KPI: 기본 성능 기준 모델 (ROC-AUC 기준)
    with c1:
        st.metric(
            "기본 성능 기준 모델",
            top_base["model"] if top_base is not None else "N/A",
            help="ROC-AUC 기준 최고 모델",
        )

    # 2번 KPI: 최적 threshold (F1 기준)
    with c2:
        st.metric(
            "최적 Threshold",
            f"{top_tuned['threshold']:.2f}" if top_tuned is not None else "N/A",
            help="F1 기준 최적 임계값",
        )

    # 3번 KPI: 최고 F1 점수
    with c3:
        st.metric(
            "최고 F1",
            f"{top_tuned['f1']:.3f}" if top_tuned is not None else "N/A",
            help="Tuning 후 최고 F1 스코어",
        )

    # 구분선 추가
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 탭 구성 ──
    # 3가지 분석 관점을 탭으로 구분
    tab1, tab2, tab3 = st.tabs(["기본 비교 (threshold 0.5)", "Tuned 비교", "Threshold 곡선"])

    # -------------------------------
    # 탭 1: 기본 모델 성능 비교
    # -------------------------------
    with tab1:
        # 데이터 없으면 안내 메시지 출력
        if base_df.empty:
            st.info("model_comparison.csv 파일을 찾을 수 없습니다.")
        else:
            # 데이터프레임 출력
            st.dataframe(base_df, use_container_width=True)
            # 설명 노트
            _note(
                "기본 threshold 0.5 적용 결과입니다. "
                "각 모델의 기본 분류 성능을 확인하는 출발점입니다."
            )

    # -------------------------------
    # 탭 2: 튜닝된 모델 성능 비교
    # -------------------------------
    with tab2:
        if tuned_df.empty:
            st.info("model_comparison_tuned.csv 파일을 찾을 수 없습니다.")
        else:
            st.dataframe(tuned_df, use_container_width=True)
            _note(
                "threshold를 조정한 뒤의 성능 비교입니다. "
                "실제 업무 목적에 맞는 운영 기준을 별도로 설정하는 것이 더 중요합니다."
            )

    # -------------------------------
    # 탭 3: threshold별 성능 변화
    # -------------------------------
    with tab3:
        if threshold_df.empty:
            st.info("threshold_metrics_all_models.csv 파일을 찾을 수 없습니다.")
        else:
            # 모델별 threshold-F1 관계를 피벗 테이블로 변환
            pivot_f1       = threshold_df.pivot(index="threshold", columns="model", values="f1")
            # 모델별 threshold-precision 관계
            pivot_precision = threshold_df.pivot(index="threshold", columns="model", values="precision")
            # 모델별 threshold-recall 관계
            pivot_recall   = threshold_df.pivot(index="threshold", columns="model", values="recall")

            # F1 그래프
            st.markdown("**F1**")
            st.line_chart(pivot_f1, use_container_width=True)
            _note("precision과 recall의 균형이 가장 잘 맞는 threshold를 찾기 위한 그래프입니다.")

            # Precision 그래프
            st.markdown("**Precision**")
            st.line_chart(pivot_precision, use_container_width=True)
            _note("threshold를 높일수록 precision은 올라가지만, 실제 이탈 고객을 놓칠 가능성도 커집니다.")

            # Recall 그래프
            st.markdown("**Recall**")
            st.line_chart(pivot_recall, use_container_width=True)
            _note("churn 문제에서는 실제 이탈 고객을 놓치지 않는 것이 중요하므로 recall 변화에 주목합니다.")

    # ── 실무 해석 ──
    # 분석 결과를 실제 비즈니스 관점으로 해석하는 영역
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 실무 해석")

    # 2개 컬럼으로 설명 영역 구성
    left, right = st.columns(2)

    # 왼쪽: threshold tuning 필요성 설명
    with left:
        st.markdown(
            """
            <div class="card-gray">
                <h4>왜 Threshold Tuning이 필요한가?</h4>
                <p style="color:#334155; font-size:0.93rem; line-height:1.75; margin:0;">
                churn 문제에서 기본값 0.5는 항상 최적이 아닙니다.
                이탈 고객을 더 많이 잡아내려면 threshold를 낮추는 전략이 필요할 수 있습니다.<br><br>
                <strong>이탈 고객 포착률(recall)</strong>과
                <strong>과탐지 방지(precision)</strong> 사이의 균형을 맞추는 작업입니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 오른쪽: 최적 모델 및 threshold 결과 요약
    with right:
        if top_tuned is not None:
            st.markdown(
                f"""
                <div class="card-gray">
                    <h4>결론</h4>
                    <p style="color:#334155; font-size:0.93rem; line-height:1.75; margin:0;">
                    <strong>{top_tuned['model']}</strong>이
                    threshold <strong>{top_tuned['threshold']:.2f}</strong>에서
                    F1 <strong>{top_tuned['f1']:.3f}</strong>으로 가장 균형 잡힌 성능을 보였습니다.<br><br>
                    운영 환경에서는 기본 0.5를 고정하기보다
                    <strong>업무 목적에 맞춰 threshold를 조정</strong>하는 접근이 적절합니다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # 튜닝 데이터 없을 경우 안내
            st.info("Tuned 결과를 불러오지 못했습니다.")