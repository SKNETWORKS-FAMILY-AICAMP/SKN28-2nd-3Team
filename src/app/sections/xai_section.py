from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from src.app.utils.load_data import load_xai_summary
from src.config.paths import XAI_OUTPUT_DIR


FEATURE_KR_MAP = {
    "active_subscription_ratio": {
        "label": "활성 구독 비율",
        "desc": "현재 보유한 구독 중 실제로 활성 상태인 구독의 비율",
        "point": "낮을수록 고객이 서비스를 실제로 덜 활용하고 있을 가능성이 있다.",
    },
    "error_rate": {
        "label": "오류 발생 비율",
        "desc": "전체 사용 또는 요청 대비 오류가 발생한 비율",
        "point": "높을수록 서비스 품질에 대한 불만이 커질 수 있다.",
    },
    "industry_DevTools": {
        "label": "산업군: 개발도구",
        "desc": "해당 고객이 DevTools 산업군에 속하는지 여부를 나타내는 변수",
        "point": "특정 산업군 고객은 사용 패턴과 이탈 요인이 다를 수 있다.",
    },
    "avg_first_response_time_minutes": {
        "label": "평균 첫 응답 시간",
        "desc": "고객 문의 발생 후 첫 답변까지 걸린 평균 시간(분)",
        "point": "길수록 고객 지원 만족도가 낮아질 가능성이 있다.",
    },
    "days_since_last_usage": {
        "label": "마지막 사용 이후 경과일",
        "desc": "고객이 마지막으로 서비스를 사용한 뒤 지난 일수",
        "point": "길수록 최근 서비스 이용이 줄어든 상태로 볼 수 있다.",
    },
    "recent_upgrade_90d": {
        "label": "최근 90일 내 업그레이드 여부",
        "desc": "최근 90일 안에 상위 요금제나 기능 업그레이드가 있었는지 여부",
        "point": "업그레이드 경험은 보통 긍정 신호지만, 맥락에 따라 기대 불일치 가능성도 있다.",
    },
    "max_sub_seats": {
        "label": "최대 구독 좌석 수",
        "desc": "해당 계정이 보유했던 최대 사용자 좌석 수",
        "point": "좌석 수 변화는 계정 규모와 서비스 활용 수준을 보여줄 수 있다.",
    },
    "error_per_subscription": {
        "label": "구독당 오류 수",
        "desc": "구독 1개당 평균적으로 발생한 오류 수",
        "point": "단순 오류율과 달리, 실제 구독 단위에서 체감되는 불편을 보여줄 수 있다.",
    },
    "health_score": {
        "label": "고객 헬스 스코어",
        "desc": "사용량, 활동성, 지원 이력 등을 종합해 만든 고객 상태 점수",
        "point": "낮을수록 이탈 위험 신호로 해석될 가능성이 높다.",
    },
    "usage_per_subscription": {
        "label": "구독당 사용량",
        "desc": "구독 수 대비 평균 서비스 사용량",
        "point": "낮을수록 구독은 유지하지만 실제 사용은 적은 상태일 수 있다.",
    },
    "avg_subscription_duration_days": {
        "label": "평균 구독 유지 기간",
        "desc": "고객이 구독을 유지한 평균 기간(일수)",
        "point": "짧을수록 서비스 정착도가 낮을 가능성이 있다.",
    },
    "total_subscriptions": {
        "label": "전체 구독 수",
        "desc": "해당 고객 계정이 보유한 전체 구독 개수",
        "point": "구독 수 자체는 규모를 보여주지만, 활성 여부와 함께 봐야 한다.",
    },
    "active_subscriptions": {
        "label": "활성 구독 수",
        "desc": "현재 실제로 사용 중인 활성 구독 개수",
        "point": "적을수록 서비스 활용 범위가 축소되고 있을 수 있다.",
    },
    "avg_mrr_amount": {
        "label": "평균 월 반복 매출(MRR)",
        "desc": "고객이 월 단위로 발생시키는 평균 반복 매출 금액",
        "point": "매출 규모는 중요하지만, 사용량과 만족도 없이 단독으로 해석하면 한계가 있다.",
    },
    "seats": {
        "label": "현재 좌석 수",
        "desc": "현재 계정이 사용 중인 사용자 좌석 수",
        "point": "좌석 수 감소는 조직 내 사용 축소 신호일 수 있다.",
    },
}


def _feature_card(label: str, value: str) -> str:
    return dedent(f"""
    <div class="mini-card">
        <div class="mini-label">{label}</div>
        <div style="
            font-size:1.15rem;
            font-weight:800;
            color:#0f172a;
            margin-top:0.55rem;
            word-break:break-word;
            overflow-wrap:anywhere;
        ">
            {value}
        </div>
    </div>
    """).strip()


def _section_card(title: str, body_html: str) -> None:
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


def _center_note(text: str) -> None:
    st.markdown(
        dedent(f"""
        <div style="
            text-align:center;
            color:#111827;
            font-size:1rem;
            line-height:1.75;
            margin-top:0.35rem;
            margin-bottom:1.1rem;
        ">
            {text}
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def _show_centered_image_with_note(
    path: Path,
    caption: str,
    note: str,
    width_ratio: float = 0.72,
) -> None:
    if not path.exists():
        st.info(f"{path.name} 파일이 없습니다.")
        return

    if width_ratio >= 0.95:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        side = (1 - width_ratio) / 2
        left, center, right = st.columns([side, width_ratio, side])
        with center:
            st.image(str(path), caption=caption, use_container_width=True)

    _center_note(note)


def _build_feature_explanation_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    df = summary.copy()
    df["중요도"] = df["mean_abs_shap"].round(4)
    df["한글 변수명"] = df["feature"].apply(
        lambda x: FEATURE_KR_MAP.get(x, {}).get("label", "설명 준비 중")
    )
    df["한글 설명"] = df["feature"].apply(
        lambda x: FEATURE_KR_MAP.get(x, {}).get("desc", "해당 변수 설명을 추가해주세요.")
    )
    df["해석 포인트"] = df["feature"].apply(
        lambda x: FEATURE_KR_MAP.get(x, {}).get("point", "이 변수의 해석 포인트를 추가해주세요.")
    )

    df = df.rename(columns={"feature": "feature 원문"})
    return df[["feature 원문", "한글 변수명", "중요도", "한글 설명", "해석 포인트"]]


def render() -> None:
    st.markdown("## 이탈 요인 해석 및 고객 유지 전략")

    summary = load_xai_summary()
    top_features = summary["feature"].head(5).tolist() if not summary.empty else []

    st.markdown(
        dedent("""
        <div class="section-card">
            <h4 style="margin-top:0; margin-bottom:0.9rem;">핵심 메시지</h4>
            <p style="line-height:1.8; margin-bottom:0;">
                설명 가능한 AI(XAI)는 모델이 높은 churn 확률을 부여한 이유를 해석하게 해준다.
                본 프로젝트에서는 주요 이탈 신호를 확인한 뒤, 이를 제품 사용성, 장애 경험,
                고객 지원 품질, 활성도 저하 같은 실무 개입 포인트와 연결했다.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            _feature_card("대표 이탈 신호 1", top_features[0] if len(top_features) > 0 else "N/A"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _feature_card("대표 이탈 신호 2", top_features[1] if len(top_features) > 1 else "N/A"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _feature_card("대표 이탈 신호 3", top_features[2] if len(top_features) > 2 else "N/A"),
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(["XAI 요약표", "SHAP 시각화", "유지 전략 제안"])

    with tab1:
        st.subheader("주요 이탈 신호")
        st.caption("상위 15개 중요 변수에 대해 원문 변수명, 한글 설명, 해석 포인트를 함께 제공합니다.")

        if summary.empty:
            st.info("xai_summary_report.csv 파일이 없습니다.")
        else:
            summary_view = _build_feature_explanation_table(summary.head(15))

            st.dataframe(
                summary_view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "feature 원문": st.column_config.TextColumn(
                        "feature 원문",
                        help="모델 학습에 실제 사용된 원본 변수명",
                        width="medium",
                    ),
                    "한글 변수명": st.column_config.TextColumn(
                        "한글 변수명",
                        help="사용자가 이해하기 쉽도록 풀어쓴 변수 이름",
                        width="medium",
                    ),
                    "중요도": st.column_config.NumberColumn(
                        "중요도",
                        help="값이 클수록 모델 예측에 더 큰 영향을 준 변수",
                        format="%.4f",
                        width="small",
                    ),
                    "한글 설명": st.column_config.TextColumn(
                        "한글 설명",
                        help="이 변수가 실제로 무엇을 뜻하는지 설명",
                        width="large",
                    ),
                    "해석 포인트": st.column_config.TextColumn(
                        "해석 포인트",
                        help="실무에서 이 변수를 어떻게 읽으면 좋은지 안내",
                        width="large",
                    ),
                },
            )

            _center_note(
                "이 표는 모델이 churn 여부를 판단할 때 중요하게 반영한 상위 15개 변수를 정리한 결과이다. "
                "원본 변수명만 보여주는 대신, 각 변수의 의미와 해석 포인트를 함께 제시해 "
                "처음 방문한 사용자도 결과를 이해할 수 있도록 구성했다."
            )

            with st.expander("처음 보는 분을 위한 해석 가이드", expanded=False):
                st.markdown(
                    """
                    **이 표를 읽는 방법**

                    - **중요도**: 숫자가 클수록 모델이 더 중요하게 본 변수이다.
                    - **한글 설명**: 변수 자체가 무엇을 의미하는지 쉽게 풀어쓴 내용이다.
                    - **해석 포인트**: 이 변수가 높거나 낮을 때 어떤 의미로 볼 수 있는지 안내한다.

                    예를 들어 `days_since_last_usage`가 중요 변수라면,
                    최근에 서비스를 오래 사용하지 않은 고객일수록 이탈 위험이 높아질 수 있다고 이해하면 된다.
                    """
                )

    with tab2:
        st.subheader("SHAP 기반 해석")

        summary_path = XAI_OUTPUT_DIR / "shap_summary.png"
        bar_path = XAI_OUTPUT_DIR / "shap_bar.png"

        _show_centered_image_with_note(
            summary_path,
            "전체 변수 영향 방향과 크기",
            "이 그래프는 각 변수가 churn 확률을 높이는 방향인지 낮추는 방향인지, 그리고 그 영향의 크기가 어느 정도인지를 함께 보여준다. "
            "즉, 모델이 어떤 조건에서 고객을 더 위험하게 판단하는지 구조적으로 해석할 수 있게 해준다.",
            width_ratio=0.72,
        )

        _show_centered_image_with_note(
            bar_path,
            "평균 영향도 기준 상위 변수",
            "이 그래프는 전체 고객을 기준으로 평균 영향력이 큰 변수를 요약한 결과이다. "
            "따라서 개별 사례를 넘어서, 어떤 변수들이 전반적으로 churn 예측에 핵심적인 역할을 했는지 파악하는 데 적합하다.",
            width_ratio=0.62,
        )

    with tab3:
        st.subheader("고객 유지 전략 제안")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            _section_card(
                "전략 1. 사용량 저하 고객 선제 케어",
                """
<p style="line-height:1.8; margin-bottom:0;">
최근 사용량 감소, 마지막 사용일 증가, 활성 구독 비율 저하는 이탈 전조로 해석할 수 있다.
따라서 로그인 빈도, 사용시간, 핵심 기능 사용량이 감소한 고객에게는
튜토리얼, 리마인드 메일, 기능 재활성화 캠페인을 우선 적용할 수 있다.
</p>
                """,
            )
        with r1c2:
            _section_card(
                "전략 2. 오류 경험 고객 즉시 대응",
                """
<p style="line-height:1.8; margin-bottom:0;">
error_rate와 응답 지연은 서비스 품질에 대한 불만을 키울 수 있다.
따라서 장애 경험이 잦은 고객에게는 우선 대응 SLA, 전담 지원,
문제 복구 후 후속 안내가 필요하다.
</p>
                """,
            )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            _section_card(
                "전략 3. 고위험 고객군 세분화 운영",
                """
<p style="line-height:1.8; margin-bottom:0;">
모든 churn 위험 고객을 동일하게 볼 수는 없다.
제품 사용 저하형, 장애 불만형, 지원 응답 불만형처럼
위험 원인을 유형화하면 훨씬 정교한 retention 액션 설계가 가능하다.
</p>
                """,
            )
        with r2c2:
            _section_card(
                "핵심 메시지",
                """
<p style="line-height:1.8; margin-bottom:0;">
모델이 고객을 왜 이탈 위험으로 판단했는지를 설명하는 단계이다.
주요 이탈 신호를 해석하고, 이를 실제 고객 유지 전략으로 연결하는 것이 목적이다.
</p>
                """,
            )