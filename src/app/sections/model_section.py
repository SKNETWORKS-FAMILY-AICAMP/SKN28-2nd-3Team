from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app.utils.load_data import (
    load_model_comparison,
    load_model_comparison_tuned,
    load_threshold_metrics_all_models,
)


def _safe_top_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    return df.sort_values(metric, ascending=False).iloc[0]


def _center_note(text: str) -> None:
    st.markdown(
        f"""
        <div style="
            text-align: center;
            color: #000000;
            font-size: 1rem;
            line-height: 1.7;
            margin-top: 0.35rem;
            margin-bottom: 1.1rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown("## 모델 성능 및 운영 기준")

    base_df = load_model_comparison()
    tuned_df = load_model_comparison_tuned()
    threshold_df = load_threshold_metrics_all_models()

    top_base = _safe_top_row(base_df, "roc_auc")
    top_tuned = _safe_top_row(tuned_df, "f1")

    c1, c2, c3 = st.columns(3)
    with c1:
        if top_base is not None:
            st.metric("기본 성능 기준 주목 모델", str(top_base["model"]))
        else:
            st.metric("기본 성능 기준 주목 모델", "N/A")

    with c2:
        if top_tuned is not None:
            st.metric("최적 threshold", f"{top_tuned['threshold']:.2f}")
        else:
            st.metric("최적 threshold", "N/A")

    with c3:
        if top_tuned is not None:
            st.metric("최고 F1", f"{top_tuned['f1']:.3f}")
        else:
            st.metric("최고 F1", "N/A")

        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin-top:0;">핵심 메시지</h4>
                <p style="line-height:1.7; margin-bottom:0;">
                어떤 모델이 좋은지보다, 어떤 기준으로 운영할지를 결정하는 단계이다.<br>
                threshold 변화에 따른 precision·recall·F1의 균형을 함께 비교하였다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(["기본 비교", "Tuned 비교", "Threshold 곡선"])

    with tab1:
        st.subheader("기본 threshold(0.5) 기준 비교")
        if base_df.empty:
            st.info("model_comparison.csv 파일이 없습니다.")
        else:
            st.dataframe(base_df, use_container_width=True)
            _center_note(
                "이 표는 기본 threshold 0.5를 적용했을 때의 모델 성능 비교 결과이다. "
                "즉, threshold를 따로 조정하지 않았을 때 각 모델이 어느 정도의 기본 분류 성능을 보이는지 확인하는 단계라고 볼 수 있다."
            )

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

    with tab3:
        st.subheader("Threshold 변화에 따른 지표 변화")
        if threshold_df.empty:
            st.info("threshold_metrics_all_models.csv 파일이 없습니다.")
        else:
            pivot_f1 = threshold_df.pivot(index="threshold", columns="model", values="f1")
            pivot_precision = threshold_df.pivot(index="threshold", columns="model", values="precision")
            pivot_recall = threshold_df.pivot(index="threshold", columns="model", values="recall")

            st.markdown("### F1 변화")
            st.line_chart(pivot_f1, use_container_width=True)
            _center_note(
                "이 그래프는 threshold 변화에 따라 F1이 어떻게 달라지는지를 보여준다. "
                "즉, precision과 recall의 균형이 가장 잘 맞는 지점을 찾기 위해 확인하는 그래프라고 해석할 수 있다."
            )

            st.markdown("### Precision 변화")
            st.line_chart(pivot_precision, use_container_width=True)
            _center_note(
                "이 그래프는 threshold 변화에 따라 precision이 어떻게 달라지는지를 보여준다. "
                "threshold를 높일수록 일반적으로 precision은 높아질 수 있지만, 그만큼 실제 이탈 고객을 놓칠 가능성도 함께 커질 수 있다."
            )

            st.markdown("### Recall 변화")
            st.line_chart(pivot_recall, use_container_width=True)
            _center_note(
                "이 그래프는 threshold 변화에 따라 recall이 어떻게 달라지는지를 보여준다. "
                "실무적으로 churn 문제에서는 실제 이탈 고객을 놓치지 않는 것이 중요하므로, recall의 변화는 운영 기준을 정할 때 특히 중요한 판단 근거가 된다."
            )

    st.markdown("### 실무 해석")
    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin-top:0;">왜 threshold tuning이 필요한가?</h4>
                <p style="line-height:1.8; margin-bottom:0;">
                churn 문제에서는 기본 0.5 기준이 항상 최적이 아니다.
                실제로는 이탈 고객을 더 많이 잡아내기 위해 threshold를 낮추는 전략이 필요할 수 있다.
                이 과정은 <b>이탈 고객 포착률(recall)</b>과 <b>불필요한 과탐지(precision 저하)</b> 사이의 균형을 맞추는 작업이다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if top_tuned is not None:
            st.markdown(
                f"""
                <div class="section-card">
                    <h4 style="margin-top:0;">결론</h4>
                    <p style="line-height:1.8; margin-bottom:0;">
                    본 프로젝트에서는 <b>{top_tuned['model']}</b>이(가)
                    threshold <b>{top_tuned['threshold']:.2f}</b>에서
                    가장 균형 잡힌 F1(<b>{top_tuned['f1']:.3f}</b>)을 보였다.
                    따라서 운영 환경에서는 기본 0.5를 고정하기보다,
                    <b>업무 목적에 맞춰 threshold를 조정하는 접근</b>이 더 적절하다고 해석할 수 있다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Tuned 결과를 불러오지 못했습니다.")