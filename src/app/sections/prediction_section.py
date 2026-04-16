# prediction_section.py

from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터 처리용 라이브러리
import streamlit as st  # Streamlit UI 구성 라이브러리

# 예측 결과 및 모델 관련 데이터 로드 함수들
from src.app.utils.load_data import (
    build_prediction_comparison,   # 모델별 예측 결과 비교 데이터 생성
    get_tuned_threshold_map,       # 모델별 최적 threshold 값 로드
    load_model_comparison,         # 기본 모델 성능 데이터
    load_model_comparison_tuned,   # 튜닝된 모델 성능 데이터
    load_x_test,                  # 테스트셋 feature 데이터
)


# 확률 값을 보기 좋게 포맷팅하는 함수
def _fmt_prob(prob) -> str:
    # 값이 없으면 N/A 반환
    if prob is None or pd.isna(prob):
        return "N/A"
    # 소수점 3자리로 포맷
    return f"{prob:.3f}"


# 확률과 threshold를 기준으로 위험 여부 라벨 생성
def _risk_label(prob, threshold) -> tuple[str, str]:
    """(label_text, tag_class)"""
    # 값이 없으면 판단 불가
    if prob is None or pd.isna(prob) or threshold is None:
        return "판단 불가", ""
    # threshold 이상이면 위험
    if prob >= threshold:
        return "위험", "tag-red"
    # 아니면 안정
    return "안정", "tag-green"


# 예측 결과에 따른 액션 가이드 생성
def _action_guide(prob, threshold) -> str:
    # 값이 없으면 안내 불가
    if prob is None or pd.isna(prob) or threshold is None:
        return "예측 결과가 없어 후속 액션을 제안하기 어렵습니다."
    # 매우 높은 위험 (threshold + 0.15 이상)
    if prob >= threshold + 0.15:
        return "즉시 케어 대상입니다. 전담 CS 연결, 사용 저해 요인 확인, 핵심 기능 재안내를 진행하세요."
    # 기준 이상 (주의 단계)
    if prob >= threshold:
        return "주의 고객입니다. 리텐션 메시지 발송, 사용량 추적, 이슈 여부 점검이 적절합니다."
    # 안정 고객
    return "현재는 안정 고객입니다. 사용량 감소나 오류율 상승 여부를 지속 모니터링하세요."


# 실제 churn 값(0/1)을 사람이 이해하기 쉽게 변환
def _actual_text(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        v = int(value)
        return "1 — 이탈" if v == 1 else "0 — 유지"
    except Exception:
        return "N/A"


# 가장 좋은 모델을 선택 (tuned → base 순)
def _find_best_model(base_df, tuned_df) -> str | None:
    # tuned 결과에서 F1 기준 최고 모델
    if not tuned_df.empty and "f1" in tuned_df.columns:
        try:
            return str(tuned_df.sort_values("f1", ascending=False).iloc[0]["model"])
        except Exception:
            pass
    # fallback: base 모델에서 ROC-AUC 기준 최고
    if not base_df.empty and "roc_auc" in base_df.columns:
        try:
            return str(base_df.sort_values("roc_auc", ascending=False).iloc[0]["model"])
        except Exception:
            pass
    return None


# note 스타일 안내 문구 출력
def _note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# -------------------------------
# 메인 렌더링 함수
# -------------------------------
def render() -> None:
    # 예측 결과 데이터 로드
    pred_df = build_prediction_comparison()

    # 테스트셋 feature 데이터 로드
    X_test = load_x_test()

    # 모델 성능 데이터 로드
    base_df = load_model_comparison()
    tuned_df = load_model_comparison_tuned()

    # 모델별 threshold 값
    threshold_map = get_tuned_threshold_map()

    # 페이지 제목
    st.markdown(
        """
        <div class="section-title">고객별 예측</div>
        <div class="section-sub">개별 고객의 이탈 위험도를 확인하고 실제 이탈 여부와 비교합니다</div>
        """,
        unsafe_allow_html=True,
    )

    # 데이터 없으면 종료
    if pred_df.empty or X_test.empty:
        st.warning("예측 비교용 데이터가 없습니다.")
        return

    # 설명 문구
    st.info(
        "**churn_flag**: 1 = 이탈 고객, 0 = 유지 고객 | "
        "Random Forest는 비교 모델로만 활용하며, 이 페이지는 **Logistic Regression** 및 **DL MLP** 결과를 기준으로 합니다."
    )

    # account_id 컬럼 확인
    account_col = "account_id" if "account_id" in pred_df.columns else None
    if account_col is None:
        st.error("account_id 컬럼이 없어 고객별 비교가 불가능합니다.")
        return

    # 고객 리스트 생성
    account_list = pred_df[account_col].dropna().astype(str).tolist()

    # 고객 선택 UI
    selected = st.selectbox("분석할 고객 선택 (account_id)", account_list)

    # 선택된 고객의 예측 결과
    row_pred = pred_df[pred_df[account_col].astype(str) == selected].head(1)

    # 선택된 고객의 feature 데이터
    row_x = X_test[X_test[account_col].astype(str) == selected].head(1) if account_col in X_test.columns else pd.DataFrame()

    # 예측 데이터 없으면 종료
    if row_pred.empty:
        st.warning("선택한 고객의 예측 결과를 찾을 수 없습니다.")
        return

    # 모델별 예측 확률
    lr_prob = row_pred["ml_logistic_proba"].iloc[0] if "ml_logistic_proba" in row_pred.columns else pd.NA
    dl_prob = row_pred["dl_mlp_proba"].iloc[0]       if "dl_mlp_proba" in row_pred.columns       else pd.NA

    # 실제 churn 값
    actual  = row_pred["actual_churn_flag"].iloc[0]  if "actual_churn_flag" in row_pred.columns  else pd.NA

    # 모델별 threshold
    lr_th = threshold_map.get("logistic_regression", 0.5)
    dl_th = threshold_map.get("DL_MLP", 0.5)

    # ── 예측 결과 ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 3개 컬럼 구성
    c1, c2, c3 = st.columns([1, 1, 0.9])

    # Logistic Regression 결과
    with c1:
        lr_label, lr_cls = _risk_label(lr_prob, lr_th)
        st.metric("Logistic Regression", _fmt_prob(lr_prob))
        st.markdown(f'<span class="tag {lr_cls}">{lr_label}</span> threshold {lr_th:.2f}', unsafe_allow_html=True)

    # DL MLP 결과
    with c2:
        dl_label, dl_cls = _risk_label(dl_prob, dl_th)
        st.metric("DL MLP", _fmt_prob(dl_prob))
        st.markdown(f'<span class="tag {dl_cls}">{dl_label}</span> threshold {dl_th:.2f}', unsafe_allow_html=True)

    # 실제 값
    with c3:
        st.metric("실제 churn_flag", _actual_text(actual))

    # ── 추천 액션 ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 최적 모델 선택
    best_model_name = _find_best_model(base_df, tuned_df)

    best_prob, best_th = None, None

    # 모델별로 적절한 확률 선택
    if best_model_name == "logistic_regression":
        best_prob, best_th = lr_prob, lr_th
    elif best_model_name == "DL_MLP":
        best_prob, best_th = dl_prob, dl_th
    else:
        # fallback
        if not pd.isna(lr_prob):
            best_model_name, best_prob, best_th = "logistic_regression", lr_prob, lr_th
        elif not pd.isna(dl_prob):
            best_model_name, best_prob, best_th = "DL_MLP", dl_prob, dl_th

    # 액션 생성
    action = _action_guide(best_prob, best_th)

    # 가운데 정렬 카드 출력
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            f"""
            <div class="card" style="text-align:center; max-width:600px; margin:auto;">
                <h4>추천 액션 — {best_model_name or 'N/A'} 기준</h4>
                <p style="color:#0f172a; font-size:1rem; line-height:1.8; margin:0;">
                    {action}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    # ── feature 미리보기 ──
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("#### 선택 고객 Feature 미리보기")

        if not row_x.empty:
            # 주요 feature 15개만 표시
            cols = [c for c in row_x.columns if c != "account_id"][:15]
            preview_cols = ([account_col] if account_col in row_x.columns else []) + cols

            st.dataframe(row_x[preview_cols], use_container_width=True)

            _note("선택 고객의 주요 feature 값입니다.")
        else:
            st.caption("원본 feature 정보를 불러오지 못했습니다.")

    # 해석 설명 카드
    with right:
        st.markdown(
            """
            <div class="card-gray">
                <h4>해석 포인트</h4>
                <p style="color:#334155; font-size:0.93rem; line-height:1.75;">
                테스트셋 고객 기준이며,
                예측과 실제 결과를 비교하여 모델 판단을 이해합니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 전체 테스트셋 ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 전체 테스트셋 예측 비교")

    # 표시할 컬럼 선택
    show_cols = [c for c in ["account_id", "ml_logistic_proba", "dl_mlp_proba", "actual_churn_flag"] if c in pred_df.columns]

    display_df = pred_df[show_cols].copy()

    # actual 값 변환
    if "actual_churn_flag" in display_df.columns:
        display_df["actual_churn_flag"] = display_df["actual_churn_flag"].apply(_actual_text)

    # 데이터 출력
    st.dataframe(display_df, use_container_width=True)

    _note("테스트셋 전체 고객에 대한 예측 비교 결과입니다.")