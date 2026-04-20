# eda_section.py

from __future__ import annotations

import streamlit as st
from src.app.utils.load_data import load_group_mean
from src.config.paths import EDA_PLOTS_DIR


def _note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def _show_image(path, caption: str, note: str, wide: bool = False) -> None:
    if not path.exists():
        return
    if wide:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        _, center, _ = st.columns([1, 2.4, 1])
        with center:
            st.image(str(path), caption=caption, use_container_width=True)
    _note(note)


def render() -> None:
    st.markdown(
        """
        <div class="section-title">EDA</div>
        <div class="section-sub">이탈 고객과 유지 고객의 차이를 비교해 이탈 신호를 탐색합니다</div>
        """,
        unsafe_allow_html=True,
    )

    # ── group mean 테이블 ──
    group_mean = load_group_mean()
    if not group_mean.empty:
        st.markdown("#### Churn 여부별 평균 차이 — 상위 변수")
        st.dataframe(group_mean.head(20), use_container_width=True)
        _note(
            "churn 고객과 유지 고객 사이에서 평균 차이가 크게 나타난 변수 순으로 정렬했습니다. "
            "차이가 클수록 이탈과 밀접하게 연결된 특성으로 해석할 수 있습니다."
        )
    else:
        st.info("group_mean_by_churn.csv 파일을 찾을 수 없습니다.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 시각화 ──
    st.markdown("#### 주요 시각화")

    specs = [
        (
            "target_distribution_overall.png",
            "전체 Churn 분포",
            "이탈 고객(22%)과 유지 고객의 비율입니다. 클래스 불균형이 있어 accuracy보다 recall·F1을 함께 평가합니다.",
            False,
        ),
        (
            "correlation_heatmap_key_features.png",
            "핵심 변수 상관관계 히트맵",
            "주요 변수들 사이의 상관관계입니다. 유사한 정보를 담는 변수 묶음이 함께 이탈 신호로 작동할 수 있습니다.",
            True,
        ),
        (
            "bar_mean_by_churn_usage.png",
            "사용량 평균 비교",
            "churn 여부에 따른 사용량 평균 차이입니다. 사용 저하는 단순 현상이 아닌 실제 이탈 위험 신호일 수 있습니다.",
            False,
        ),
        (
            "bar_mean_by_churn_health_score.png",
            "Health Score 평균 비교",
            "고객 상태 종합 지표인 health score의 churn 여부별 차이입니다. 낮을수록 이탈 위험이 높게 나타납니다.",
            False,
        ),
    ]

    for filename, caption, note, wide in specs:
        path = EDA_PLOTS_DIR / filename
        _show_image(path, caption, note, wide=wide)