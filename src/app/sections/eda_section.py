from __future__ import annotations

import streamlit as st
from src.app.utils.load_data import load_group_mean
from src.config.paths import EDA_PLOTS_DIR


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


def _show_centered_image_with_note(
    path,
    caption: str,
    note: str,
    big: bool = False,
) -> None:
    if not path.exists():
        return

    if big:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        left, center, right = st.columns([1, 2.2, 1])
        with center:
            st.image(str(path), caption=caption, use_container_width=True)

    _center_note(note)


def render() -> None:
    st.markdown("## EDA: 이탈 고객 특성 탐색")

    st.markdown(
        """
        <div class="section-card">
            <h4 style="margin-top:0;">목적</h4>
            <p style="line-height:1.7; margin-bottom:0;">
            churn 고객과 유지 고객의 차이를 비교하여, 
            어떤 특성이 이탈과 연결되는지 빠르게 파악하는 단계이다.<br>
            이후 모델링과 해석의 출발점이 되는 기초 탐색 과정으로 볼 수 있다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    group_mean = load_group_mean()
    if not group_mean.empty:
        st.subheader("churn 여부별 평균 차이 상위 변수")
        st.dataframe(group_mean.head(20), use_container_width=True)
        _center_note(
            "이 표는 churn 고객과 유지 고객 사이에서 평균 차이가 크게 나타난 변수를 정리한 결과이다. "
            "즉, 어떤 특성이 이탈과 더 밀접하게 연결되어 있는지 1차적으로 파악하는 데 의미가 있다."
        )
    else:
        st.info("group_mean_by_churn.csv 파일이 없습니다.")

    st.subheader("주요 시각화")

    image_specs = [
        (
            "target_distribution_overall.png",
            "전체 churn 분포",
            "이 그래프는 전체 데이터에서 이탈 고객과 유지 고객이 어떻게 분포하는지 보여준다. "
            "클래스 불균형 정도를 확인하는 단계이며, 이후 모델 평가에서 accuracy보다 recall과 F1을 함께 봐야 하는 이유와도 연결된다.",
            False,
        ),
        (
            "correlation_heatmap_key_features.png",
            "핵심 변수 상관관계",
            "이 히트맵은 주요 변수들 사이의 상관관계를 보여준다. "
            "서로 유사한 정보를 담는 변수가 많은지 확인할 수 있으며, 해석 단계에서는 특정 변수 하나보다 변수 묶음이 함께 작동할 가능성을 시사한다.",
            True,
        ),
        (
            "bar_mean_by_churn_usage.png",
            "사용량 평균 비교",
            "이 그래프는 churn 여부에 따라 고객 사용량 평균이 어떻게 달라지는지 보여준다. "
            "이탈 고객의 사용량이 더 낮게 나타난다면, 사용 저하가 단순 현상이 아니라 실제 이탈 위험 신호일 가능성이 높다고 해석할 수 있다.",
            False,
        ),
        (
            "bar_mean_by_churn_health_score.png",
            "health score 평균 비교",
            "이 그래프는 고객 상태를 종합적으로 나타내는 health score가 churn 여부에 따라 어떻게 달라지는지 보여준다. "
            "이탈 고객의 health score가 낮다면, 서비스 전반의 품질 저하나 고객 경험 악화가 이탈과 연결될 수 있음을 시사한다.",
            False,
        ),
    ]

    for filename, caption, note, big in image_specs:
        path = EDA_PLOTS_DIR / filename
        _show_centered_image_with_note(path, caption, note, big=big)