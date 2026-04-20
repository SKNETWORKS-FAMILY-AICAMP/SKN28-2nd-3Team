# model_section.py
from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터프레임 처리용 라이브러리
import streamlit as st  # Streamlit UI 구성 라이브러리

# 모델 성능 관련 데이터를 로드하는 함수들
from src.app.utils.load_data import (
    load_model_comparison,              # 기본 모델 성능 비교 데이터 로드
    load_model_comparison_tuned,        # threshold 튜닝된 모델 성능 데이터 로드
    load_threshold_metrics_all_models,  # threshold별 성능 변화 데이터 로드
)


# 특정 metric 기준으로 가장 높은 성능을 가진 행을 반환하는 함수
def _safe_top_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    # 데이터가 없거나 metric 컬럼이 없으면 None 반환
    if df.empty or metric not in df.columns:
        return None
    # metric 기준으로 내림차순 정렬 후 최상단(최고 성능) 행 반환
    return df.sort_values(metric, ascending=False).iloc[0]


# 중앙 정렬된 설명 문구(note)를 출력하는 함수
def _center_note(text: str) -> None:
    st.markdown(
        f"""
        <div style="
            text-align: center;              /* 가운데 정렬 */
            color: #000000;                  /* 글자 색 */
            font-size: 1rem;                 /* 글자 크기 */
            line-height: 1.7;                /* 줄 간격 */
            margin-top: 0.35rem;             /* 위쪽 여백 */
            margin-bottom: 1.1rem;           /* 아래쪽 여백 */
        ">
            {text}                           
        </div>
        """,
        unsafe_allow_html=True,  # HTML 스타일 적용 허용
    )


# Streamlit에서 해당 페이지를 렌더링하는 메인 함수
def render() -> None:
    # 페이지 제목 출력
    st.markdown("## 모델 성능 및 운영 기준")

    # 데이터 로드
    base_df = load_model_comparison()          # 기본 모델 성능 데이터
    tuned_df = load_model_comparison_tuned()   # 튜닝된 모델 성능 데이터
    threshold_df = load_threshold_metrics_all_models()  # threshold별 성능 데이터

    # 기본 모델 중 ROC-AUC 기준 최고 모델 선택
    top_base = _safe_top_row(base_df, "roc_auc")

    # 튜닝된 모델 중 F1 기준 최고 모델 선택
    top_tuned = _safe_top_row(tuned_df, "f1")

    # KPI 영역: 3개 컬럼으로 주요 지표 시각화
    c1, c2, c3 = st.columns(3)

    # 1번 KPI: 기본 성능 기준 모델
    with c1:
        if top_base is not None:
            st.metric("기본 성능 기준 주목 모델", str(top_base["model"]))
        else:
            st.metric("기본 성능 기준 주목 모델", "N/A")

    # 2번 KPI: 최적 threshold
    with c2:
        if top_tuned is not None:
            st.metric("최적 threshold", f"{top_tuned['threshold']:.2f}")
        else:
            st.metric("최적 threshold", "N/A")

    # 3번 KPI: 최고 F1 점수
    with c3:
        if top_tuned is not None:
            st.metric("최고 F1", f"{top_tuned['f1']:.3f}")
        else:
            st.metric("최고 F1", "N/A")

        # 핵심 메시지 카드 출력
        st.markdown(
            """
            <div class="card-gray">
                <h4 style="margin-top:0;">핵심 메시지</h4>
                <p style="line-height:1.8; margin-bottom:0;">
                어떤 모델이 좋은지보다, 어떤 기준으로 운영할지를 결정하는 단계이다.<br>
                threshold 변화에 따른 precision·recall·F1의 균형을 함께 비교하였다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 탭 구성 (3가지 관점)
    tab1, tab2, tab3 = st.tabs(["기본 비교", "Tuned 비교", "Threshold 곡선"])

    # -------------------------------
    # 탭 1: 기본 threshold(0.5) 기준 성능 비교
    # -------------------------------
    with tab1:
        st.subheader("기본 threshold(0.5) 기준 비교")

        # 데이터 없을 경우 안내
        if base_df.empty:
            st.info("model_comparison.csv 파일이 없습니다.")
        else:
            # 데이터프레임 출력
            st.dataframe(base_df, use_container_width=True)

            # 설명 노트 출력
            _center_note(
                "이 표는 기본 threshold 0.5를 적용했을 때의 모델 성능 비교 결과이다. "
                "즉, threshold를 따로 조정하지 않았을 때 각 모델이 어느 정도의 기본 분류 성능을 보이는지 확인하는 단계라고 볼 수 있다."
            )

    # -------------------------------
    # 탭 2: threshold tuning 이후 성능 비교
    # -------------------------------
    with tab2:
        st.subheader("Threshold tuning 이후 비교")

        if tuned_df.empty:
            st.info("model_comparison_tuned.csv 파일이 없습니다.")
        else:
            st.dataframe(tuned_df, use_container_width=True)

            _center_note(
                "이 표는 threshold를 조정한 뒤의 성능 비교 결과이다. "
                "기본값 0.5를 그대로 사용하는 것보다, 실제 업무 목적에 맞는 운영 기준을 별도로 찾는 것이 더 중요하다는 점을 보여준다."
            )

    # -------------------------------
    # 탭 3: threshold 변화에 따른 성능 변화
    # -------------------------------
    with tab3:
        st.subheader("Threshold 변화에 따른 지표 변화")

        if threshold_df.empty:
            st.info("threshold_metrics_all_models.csv 파일이 없습니다.")
        else:
            # threshold 기준으로 모델별 F1 값 피벗 테이블 생성
            pivot_f1 = threshold_df.pivot(index="threshold", columns="model", values="f1")

            # threshold 기준 precision 피벗 테이블
            pivot_precision = threshold_df.pivot(index="threshold", columns="model", values="precision")

            # threshold 기준 recall 피벗 테이블
            pivot_recall = threshold_df.pivot(index="threshold", columns="model", values="recall")

            # F1 변화 그래프
            st.markdown("### F1 변화")
            st.line_chart(pivot_f1, use_container_width=True)
            _center_note(
                "이 그래프는 threshold 변화에 따라 F1이 어떻게 달라지는지를 보여준다. "
                "즉, precision과 recall의 균형이 가장 잘 맞는 지점을 찾기 위해 확인하는 그래프라고 해석할 수 있다."
            )

            # Precision 변화 그래프
            st.markdown("### Precision 변화")
            st.line_chart(pivot_precision, use_container_width=True)
            _center_note(
                "이 그래프는 threshold 변화에 따라 precision이 어떻게 달라지는지를 보여준다. "
                "threshold를 높일수록 일반적으로 precision은 높아질 수 있지만, 그만큼 실제 이탈 고객을 놓칠 가능성도 함께 커질 수 있다."
            )

            # Recall 변화 그래프
            st.markdown("### Recall 변화")
            st.line_chart(pivot_recall, use_container_width=True)
            _center_note(
                "이 그래프는 threshold 변화에 따라 recall이 어떻게 달라지는지를 보여준다. "
                "실무적으로 churn 문제에서는 실제 이탈 고객을 놓치지 않는 것이 중요하므로, recall의 변화는 운영 기준을 정할 때 특히 중요한 판단 근거가 된다."
            )

    # 구분선 추가
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 실무 해석 섹션 제목
    st.markdown("### 실무 해석")

    # 2개 컬럼 레이아웃
    left, right = st.columns(2)

    # 왼쪽: threshold tuning 필요성 설명
    with left:
        st.markdown(
            """
            <div class="card-gray">
                <h4>왜 Threshold Tuning이 필요한가?</h4>
                <p style="line-height:1.8; margin-bottom:0;">
                churn 문제에서는 기본 0.5 기준이 항상 최적이 아니다.<br><br>
                이탈 고객을 더 많이 잡아내기 위해 threshold를 낮추는 전략이 필요하다.<br><br>
                <b>recall(이탈 포착)</b>과 <b>precision(과탐 방지)</b> 사이의 균형을 맞추는 것이 핵심이다.
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
                    <p style="line-height:1.8; margin-bottom:0;">
                    <b>{top_tuned['model']}</b> 모델이<br>
                    threshold <b>{top_tuned['threshold']:.2f}</b>에서<br>
                    F1 <b>{top_tuned['f1']:.3f}</b>로 가장 균형 잡힌 성능을 보였다.<br><br>
                    따라서 운영에서는 <b>threshold를 목적에 맞게 조정</b>하는 것이 중요하다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Tuned 결과를 불러오지 못했습니다.")