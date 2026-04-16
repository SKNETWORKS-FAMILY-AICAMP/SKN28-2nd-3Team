# xai_section.py

from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

from pathlib import Path  # 파일 경로 객체를 다루기 위한 모듈
from textwrap import dedent  # 여러 줄 문자열의 공통 들여쓰기를 정리할 때 사용하는 유틸 (현재는 사용되지 않지만 확장 대비)

import pandas as pd  # 데이터프레임 처리용 라이브러리
import streamlit as st  # Streamlit UI 구성 라이브러리

from src.app.utils.load_data import load_xai_summary  # XAI 요약 결과 CSV를 불러오는 함수
from src.config.paths import XAI_OUTPUT_DIR  # SHAP 이미지가 저장된 출력 디렉토리 경로


# 원문 feature명을 한글 변수명, 설명, 해석 포인트로 연결하는 매핑 사전
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


# note 클래스 스타일을 이용해 설명 문구를 출력하는 함수
def _note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# 상위 이탈 신호를 stat-card 스타일 카드 HTML로 만들어 반환하는 함수
def _signal_card(rank: str, feature: str) -> str:
    # feature에 대한 한글 정보 조회
    info = FEATURE_KR_MAP.get(feature, {})
    # 매핑 정보가 있으면 한글명, 없으면 원문 그대로 사용
    kr_name = info[0] if info else feature
    # 카드 형태 HTML 문자열 반환
    return f"""
    <div class="stat-card">
        <div class="stat-label">{rank}</div>
        <div style="font-size:1rem; font-weight:700; color:#0f172a; margin-top:0.35rem;">{kr_name}</div>
        <div class="stat-sub" style="font-family:monospace;">{feature}</div>
    </div>
    """


# XAI 요약 데이터를 한글 설명 테이블 형태로 가공하는 함수
def _build_table(summary: pd.DataFrame) -> pd.DataFrame:
    # 원본 훼손을 막기 위해 복사본 생성
    df = summary.copy()

    # 중요도 컬럼 생성 (소수점 4자리 반올림)
    df["중요도"] = df["mean_abs_shap"].round(4)

    # feature명을 한글 변수명으로 변환
    df["한글 변수명"] = df["feature"].apply(lambda x: FEATURE_KR_MAP.get(x, ("설명 준비 중",))[0])

    # feature 설명 추가
    df["설명"] = df["feature"].apply(lambda x: FEATURE_KR_MAP.get(x, ("", "해당 변수 설명을 추가해주세요."))[1] if len(FEATURE_KR_MAP.get(x, ())) > 1 else "")

    # 해석 포인트 추가
    df["해석 포인트"] = df["feature"].apply(lambda x: FEATURE_KR_MAP.get(x, ("", "", "해석 포인트를 추가해주세요."))[2] if len(FEATURE_KR_MAP.get(x, ())) > 2 else "")

    # 원본 feature 컬럼명을 보기 좋게 변경
    df = df.rename(columns={"feature": "feature (원문)"})

    # 필요한 컬럼만 순서대로 반환
    return df[["feature (원문)", "한글 변수명", "중요도", "설명", "해석 포인트"]]


# 이미지 파일을 화면에 보여주고, 아래에 note 설명까지 출력하는 함수
def _show_image(path: Path, caption: str, note: str, ratio: float = 0.72) -> None:
    # 이미지 파일이 없으면 안내 후 종료
    if not path.exists():
        st.info(f"{path.name} 파일을 찾을 수 없습니다.")
        return

    # ratio가 거의 1이면 전체 폭으로 표시
    if ratio >= 0.95:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        # 가운데 정렬을 위해 좌우 여백 컬럼 생성
        side = (1 - ratio) / 2
        _, center, _ = st.columns([side, ratio, side])
        with center:
            st.image(str(path), caption=caption, use_container_width=True)

    # 이미지 설명 note 출력
    _note(note)


# 유지 전략 카드를 HTML 형태로 출력하는 함수
def _strategy_card(title: str, body: str, tag: str = "") -> None:
    # tag가 있으면 빨간 태그 배지 생성
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


# Streamlit에서 실제로 호출되는 XAI 섹션 메인 렌더링 함수
def render() -> None:
    # XAI 요약 결과 로드
    summary = load_xai_summary()

    # 상위 5개 중요 변수 목록 추출
    top_features = summary["feature"].head(5).tolist() if not summary.empty else []

    # 페이지 제목과 부제목 출력
    st.markdown(
        """
        <div class="section-title">이탈 요인 해석 & 유지 전략</div>
        <div class="section-sub">SHAP 기반으로 이탈 신호를 해석하고 실행 가능한 전략을 제안합니다</div>
        """,
        unsafe_allow_html=True,
    )

    # ── 상위 이탈 신호 3개 ──
    # 가장 중요한 이탈 신호 3개를 카드 형태로 보여주는 영역
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(_signal_card("대표 이탈 신호 #1", top_features[0] if len(top_features) > 0 else "N/A"), unsafe_allow_html=True)

    with c2:
        st.markdown(_signal_card("대표 이탈 신호 #2", top_features[1] if len(top_features) > 1 else "N/A"), unsafe_allow_html=True)

    with c3:
        st.markdown(_signal_card("대표 이탈 신호 #3", top_features[2] if len(top_features) > 2 else "N/A"), unsafe_allow_html=True)

    # 구분선
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 탭 구성 ──
    # XAI 요약표 / SHAP 시각화 / 유지 전략 3개 관점으로 구분
    tab1, tab2, tab3 = st.tabs(["XAI 요약표", "SHAP 시각화", "유지 전략"])

    # -------------------------------
    # 탭 1: XAI 요약표
    # -------------------------------
    with tab1:
        # 요약 데이터가 없으면 안내
        if summary.empty:
            st.info("xai_summary_report.csv 파일을 찾을 수 없습니다.")
        else:
            # 표 설명
            st.caption("상위 15개 중요 변수의 원문명, 한글 설명, 해석 포인트를 함께 제공합니다.")

            # 상위 15개 변수만 한글 해석 테이블로 변환
            view = _build_table(summary.head(15))

            # 데이터프레임 출력
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

            # 표 해설 note
            _note(
                "모델이 churn 판단 시 중요하게 반영한 상위 15개 변수입니다. "
                "중요도 값이 클수록 예측에 더 큰 영향을 준 변수입니다."
            )

            # 초심자용 읽는 법 설명
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

    # -------------------------------
    # 탭 2: SHAP 시각화
    # -------------------------------
    with tab2:
        # SHAP summary plot 이미지 출력
        _show_image(
            XAI_OUTPUT_DIR / "shap_summary.png",
            "변수 영향 방향과 크기",
            "각 변수가 churn 확률을 높이는 방향인지 낮추는 방향인지, 영향의 크기를 함께 보여줍니다.",
            ratio=0.72,
        )

        # SHAP bar plot 이미지 출력
        _show_image(
            XAI_OUTPUT_DIR / "shap_bar.png",
            "평균 영향도 기준 상위 변수",
            "전체 고객 기준 평균 영향력이 큰 변수 요약입니다. 어떤 변수가 전반적으로 중요한지 파악할 수 있습니다.",
            ratio=0.62,
        )

    # -------------------------------
    # 탭 3: 유지 전략
    # -------------------------------
    with tab3:
        # 첫 번째 줄: 전략 카드 2개
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

        # 두 번째 줄: 전략 카드 1개 + 핵심 메시지 카드 1개
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