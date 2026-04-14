
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from src.app.utils.load_data import load_xai_summary
from src.config.paths import XAI_OUTPUT_DIR


FEATURE_KR_MAP = {
    "active_subscription_ratio":           ("활성 구독 비율",        "현재 구독 중 실제로 활성 상태인 비율",          "낮을수록 서비스를 실제로 덜 활용 중일 수 있습니다."),
    "error_rate":                          ("오류 발생 비율",         "전체 사용 대비 오류 발생 비율",                 "높을수록 서비스 품질 불만이 커질 수 있습니다."),
    "industry_DevTools":                   ("산업군: 개발도구",        "DevTools 산업군 여부",                         "특정 산업군은 이탈 패턴이 다를 수 있습니다."),
    "avg_first_response_time_minutes":     ("평균 첫 응답 시간",      "문의 후 첫 답변까지 평균 시간(분)",              "길수록 고객 지원 만족도가 낮아질 수 있습니다."),
    "days_since_last_usage":               ("마지막 사용 경과일",      "마지막으로 서비스를 사용한 뒤 지난 일수",        "길수록 최근 서비스 이용이 줄어든 상태입니다."),
    "recent_upgrade_90d":                  ("최근 90일 업그레이드",   "최근 90일 내 상위 플랜 업그레이드 여부",          "업그레이드 경험은 보통 긍정 신호입니다."),
    "max_sub_seats":                       ("최대 구독 좌석 수",       "보유했던 최대 사용자 좌석 수",                  "좌석 수 변화는 계정 규모를 보여줍니다."),
    "error_per_subscription":              ("구독당 오류 수",          "구독 1개당 평균 오류 수",                       "실제 체감 불편을 보여주는 지표입니다."),
    "health_score":                        ("고객 헬스 스코어",        "사용량·활동성·지원 이력 종합 점수",              "낮을수록 이탈 위험 신호로 해석됩니다."),
    "usage_per_subscription":              ("구독당 사용량",           "구독 수 대비 평균 사용량",                       "낮을수록 구독은 유지하지만 실사용이 적은 상태입니다."),
    "avg_subscription_duration_days":      ("평균 구독 유지 기간",     "구독을 유지한 평균 기간(일)",                    "짧을수록 서비스 정착도가 낮을 수 있습니다."),
    "total_subscriptions":                 ("전체 구독 수",            "보유한 전체 구독 개수",                          "규모 지표이지만 활성 여부와 함께 해석해야 합니다."),
    "active_subscriptions":                ("활성 구독 수",            "현재 사용 중인 구독 개수",                       "적을수록 서비스 활용 범위가 축소 중일 수 있습니다."),
    "avg_mrr_amount":                      ("평균 월 반복 매출",       "월 단위 평균 반복 매출(MRR)",                    "매출 규모는 사용량·만족도와 함께 해석해야 합니다."),
    "seats":                               ("현재 좌석 수",            "현재 사용 중인 사용자 좌석 수",                  "좌석 감소는 조직 내 사용 축소 신호입니다."),
}


def _note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def _signal_card(rank: str, feature: str) -> str:
    info = FEATURE_KR_MAP.get(feature, {})
    kr_name = info[0] if info else feature
    return f"""
    <div class="stat-card">
        <div class="stat-label">{rank}</div>
        <div style="font-size:1rem; font-weight:700; color:#0f172a; margin-top:0.35rem;">{kr_name}</div>
        <div class="stat-sub" style="font-family:monospace;">{feature}</div>
    </div>
    """


def _build_table(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["중요도"] = df["mean_abs_shap"].round(4)
    df["한글 변수명"] = df["feature"].apply(lambda x: FEATURE_KR_MAP.get(x, ("설명 준비 중",))[0])
    df["설명"] = df["feature"].apply(lambda x: FEATURE_KR_MAP.get(x, ("", "해당 변수 설명을 추가해주세요."))[1] if len(FEATURE_KR_MAP.get(x, ())) > 1 else "")
    df["해석 포인트"] = df["feature"].apply(lambda x: FEATURE_KR_MAP.get(x, ("", "", "해석 포인트를 추가해주세요."))[2] if len(FEATURE_KR_MAP.get(x, ())) > 2 else "")
    df = df.rename(columns={"feature": "feature (원문)"})
    return df[["feature (원문)", "한글 변수명", "중요도", "설명", "해석 포인트"]]


def _show_image(path: Path, caption: str, note: str, ratio: float = 0.72) -> None:
    if not path.exists():
        st.info(f"{path.name} 파일을 찾을 수 없습니다.")
        return
    if ratio >= 0.95:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        side = (1 - ratio) / 2
        _, center, _ = st.columns([side, ratio, side])
        with center:
            st.image(str(path), caption=caption, use_container_width=True)
    _note(note)


def _strategy_card(title: str, body: str, tag: str = "") -> None:
    tag_html = f'<span class="tag tag-red">{tag}</span><br>' if tag else ""
    st.markdown(
        f"""
        <div class="card">
            <h4>{title}</h4>
            {tag_html}
            <p style="color:#334155; font-size:0.93rem; line-height:1.75; margin:0;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    summary = load_xai_summary()
    top_features = summary["feature"].head(5).tolist() if not summary.empty else []

    st.markdown(
        """
        <div class="section-title">이탈 요인 해석 & 유지 전략</div>
        <div class="section-sub">SHAP 기반으로 이탈 신호를 해석하고 실행 가능한 전략을 제안합니다</div>
        """,
        unsafe_allow_html=True,
    )

    # ── 상위 이탈 신호 3개 ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_signal_card("대표 이탈 신호 #1", top_features[0] if len(top_features) > 0 else "N/A"), unsafe_allow_html=True)
    with c2:
        st.markdown(_signal_card("대표 이탈 신호 #2", top_features[1] if len(top_features) > 1 else "N/A"), unsafe_allow_html=True)
    with c3:
        st.markdown(_signal_card("대표 이탈 신호 #3", top_features[2] if len(top_features) > 2 else "N/A"), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 탭 ──
    tab1, tab2, tab3 = st.tabs(["XAI 요약표", "SHAP 시각화", "유지 전략"])

    with tab1:
        if summary.empty:
            st.info("xai_summary_report.csv 파일을 찾을 수 없습니다.")
        else:
            st.caption("상위 15개 중요 변수의 원문명, 한글 설명, 해석 포인트를 함께 제공합니다.")
            view = _build_table(summary.head(15))
            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "feature (원문)": st.column_config.TextColumn(width="medium"),
                    "한글 변수명":    st.column_config.TextColumn(width="medium"),
                    "중요도":        st.column_config.NumberColumn(format="%.4f", width="small"),
                    "설명":          st.column_config.TextColumn(width="large"),
                    "해석 포인트":   st.column_config.TextColumn(width="large"),
                },
            )
            _note(
                "모델이 churn 판단 시 중요하게 반영한 상위 15개 변수입니다. "
                "중요도 값이 클수록 예측에 더 큰 영향을 준 변수입니다."
            )

            with st.expander("처음 보시는 분을 위한 읽는 법", expanded=False):
                st.markdown(
                    """
                    - **중요도**: 숫자가 클수록 모델이 더 중요하게 본 변수입니다.
                    - **설명**: 변수가 실제로 무엇을 의미하는지 풀어쓴 내용입니다.
                    - **해석 포인트**: 값이 높거나 낮을 때 어떤 의미인지 안내합니다.

                    예시 — `days_since_last_usage`가 중요하다면,
                    최근 서비스를 오래 사용하지 않은 고객일수록 이탈 위험이 높다고 이해하면 됩니다.
                    """
                )

    with tab2:
        _show_image(
            XAI_OUTPUT_DIR / "shap_summary.png",
            "변수 영향 방향과 크기",
            "각 변수가 churn 확률을 높이는 방향인지 낮추는 방향인지, 영향의 크기를 함께 보여줍니다.",
            ratio=0.72,
        )
        _show_image(
            XAI_OUTPUT_DIR / "shap_bar.png",
            "평균 영향도 기준 상위 변수",
            "전체 고객 기준 평균 영향력이 큰 변수 요약입니다. 어떤 변수가 전반적으로 중요한지 파악할 수 있습니다.",
            ratio=0.62,
        )

    with tab3:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            _strategy_card(
                "전략 1. 사용량 저하 고객 선제 케어",
                "사용량 감소·마지막 사용일 증가·활성 구독 비율 저하는 이탈 전조 신호입니다. "
                "로그인 빈도, 핵심 기능 사용량이 감소한 고객에게 "
                "튜토리얼·리마인드 메일·기능 재활성화 캠페인을 우선 적용하세요.",
                tag="사용 저하",
            )
        with r1c2:
            _strategy_card(
                "전략 2. 오류 경험 고객 즉시 대응",
                "error_rate 상승과 응답 지연은 서비스 품질 불만을 키웁니다. "
                "장애 경험이 잦은 고객에게는 우선 대응 SLA·전담 지원·"
                "문제 복구 후 후속 안내가 필요합니다.",
                tag="오류·응답 지연",
            )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            _strategy_card(
                "전략 3. 고위험 고객군 세분화 운영",
                "모든 churn 위험 고객을 동일하게 볼 수 없습니다. "
                "사용 저하형·장애 불만형·응답 불만형으로 유형화하면 "
                "훨씬 정교한 retention 액션 설계가 가능합니다.",
                tag="세분화",
            )
        with r2c2:
            st.markdown(
                """
                <div class="card-gray">
                    <h4>핵심 메시지</h4>
                    <p style="color:#334155; font-size:0.93rem; line-height:1.75; margin:0;">
                    모델이 고객을 왜 이탈 위험으로 판단했는지 설명하는 단계입니다.<br><br>
                    주요 이탈 신호를 해석하고,
                    이를 <strong>실제 고객 유지 전략으로 연결</strong>하는 것이 목적입니다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
