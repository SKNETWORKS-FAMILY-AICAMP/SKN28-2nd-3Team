from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app.utils.load_data import (
    build_prediction_comparison,
    get_tuned_threshold_map,
    load_model_comparison,
    load_model_comparison_tuned,
    load_x_test,
)


def _fmt_prob(prob) -> str:
    if prob is None or pd.isna(prob):
        return "N/A"
    return f"{prob:.3f}"


def _risk_label(prob, threshold) -> tuple[str, str]:
    """(label_text, tag_class)"""
    if prob is None or pd.isna(prob) or threshold is None:
        return "판단 불가", ""
    if prob >= threshold:
        return "위험", "tag-red"
    return "안정", "tag-green"


def _action_guide(prob, threshold) -> str:
    if prob is None or pd.isna(prob) or threshold is None:
        return "예측 결과가 없어 후속 액션을 제안하기 어렵습니다."
    if prob >= threshold + 0.15:
        return "즉시 케어 대상입니다. 전담 CS 연결, 사용 저해 요인 확인, 핵심 기능 재안내를 진행하세요."
    if prob >= threshold:
        return "주의 고객입니다. 리텐션 메시지 발송, 사용량 추적, 이슈 여부 점검이 적절합니다."
    return "현재는 안정 고객입니다. 사용량 감소나 오류율 상승 여부를 지속 모니터링하세요."


def _actual_text(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        v = int(value)
        return "1 — 이탈" if v == 1 else "0 — 유지"
    except Exception:
        return "N/A"


def _find_best_model(base_df, tuned_df) -> str | None:
    if not tuned_df.empty and "f1" in tuned_df.columns:
        try:
            return str(tuned_df.sort_values("f1", ascending=False).iloc[0]["model"])
        except Exception:
            pass
    if not base_df.empty and "roc_auc" in base_df.columns:
        try:
            return str(base_df.sort_values("roc_auc", ascending=False).iloc[0]["model"])
        except Exception:
            pass
    return None


def _note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def render() -> None:
    pred_df = build_prediction_comparison()
    X_test = load_x_test()
    base_df = load_model_comparison()
    tuned_df = load_model_comparison_tuned()
    threshold_map = get_tuned_threshold_map()

    st.markdown(
        """
        <div class="section-title">고객별 예측</div>
        <div class="section-sub">개별 고객의 이탈 위험도를 확인하고 실제 이탈 여부와 비교합니다</div>
        """,
        unsafe_allow_html=True,
    )

    if pred_df.empty or X_test.empty:
        st.warning("예측 비교용 데이터가 없습니다.")
        return

    st.info(
        "**churn_flag**: 1 = 이탈 고객, 0 = 유지 고객 | "
        "Random Forest는 비교 모델로만 활용하며, 이 페이지는 **Logistic Regression** 및 **DL MLP** 결과를 기준으로 합니다."
    )

    account_col = "account_id" if "account_id" in pred_df.columns else None
    if account_col is None:
        st.error("account_id 컬럼이 없어 고객별 비교가 불가능합니다.")
        return

    account_list = pred_df[account_col].dropna().astype(str).tolist()
    selected = st.selectbox("분석할 고객 선택 (account_id)", account_list)

    row_pred = pred_df[pred_df[account_col].astype(str) == selected].head(1)
    row_x = X_test[X_test[account_col].astype(str) == selected].head(1) if account_col in X_test.columns else pd.DataFrame()

    if row_pred.empty:
        st.warning("선택한 고객의 예측 결과를 찾을 수 없습니다.")
        return

    lr_prob = row_pred["ml_logistic_proba"].iloc[0] if "ml_logistic_proba" in row_pred.columns else pd.NA
    dl_prob = row_pred["dl_mlp_proba"].iloc[0]       if "dl_mlp_proba" in row_pred.columns       else pd.NA
    actual  = row_pred["actual_churn_flag"].iloc[0]  if "actual_churn_flag" in row_pred.columns  else pd.NA

    lr_th = threshold_map.get("logistic_regression", 0.5)
    dl_th = threshold_map.get("DL_MLP", 0.5)

    # ── 예측 결과 3개 ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 0.9])
    with c1:
        lr_label, lr_cls = _risk_label(lr_prob, lr_th)
        st.metric("Logistic Regression", _fmt_prob(lr_prob), help="선형 기준 churn 확률")
        st.markdown(f'<span class="tag {lr_cls}">{lr_label}</span> threshold {lr_th:.2f}', unsafe_allow_html=True)
    with c2:
        dl_label, dl_cls = _risk_label(dl_prob, dl_th)
        st.metric("DL MLP", _fmt_prob(dl_prob), help="비선형 상호작용 반영 churn 확률")
        st.markdown(f'<span class="tag {dl_cls}">{dl_label}</span> threshold {dl_th:.2f}', unsafe_allow_html=True)
    with c3:
        st.metric("실제 churn_flag", _actual_text(actual))

    # ── 추천 액션 ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    best_model_name = _find_best_model(base_df, tuned_df)
    best_prob, best_th = None, None
    if best_model_name == "logistic_regression":
        best_prob, best_th = lr_prob, lr_th
    elif best_model_name == "DL_MLP":
        best_prob, best_th = dl_prob, dl_th
    else:
        if not pd.isna(lr_prob):
            best_model_name, best_prob, best_th = "logistic_regression", lr_prob, lr_th
        elif not pd.isna(dl_prob):
            best_model_name, best_prob, best_th = "DL_MLP", dl_prob, dl_th

    action = _action_guide(best_prob, best_th)
    # 가운데 정렬용 컬럼
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            f"""
            <div class="card" 
                style="
                    text-align:center;
                    max-width:600px;
                    margin:auto;
                ">
                <h4>추천 액션 — {best_model_name or 'N/A'} 기준</h4>
                <p style="color:#0f172a; font-size:1rem; line-height:1.8; margin:0;">
                    {action}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── feature 미리보기 + 해석 포인트 ──
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### 선택 고객 Feature 미리보기")
        if not row_x.empty:
            cols = [c for c in row_x.columns if c != "account_id"][:15]
            preview_cols = ([account_col] if account_col in row_x.columns else []) + cols
            st.dataframe(row_x[preview_cols], use_container_width=True)
            _note("선택 고객의 주요 feature 값입니다. 예측 수치와 함께 보면 어떤 특성을 가진 고객인지 파악할 수 있습니다.")
        else:
            st.caption("원본 feature 정보를 불러오지 못했습니다.")

    with right:
        st.markdown(
            """
            <div class="card-gray">
                <h4>해석 포인트</h4>
                <p style="color:#334155; font-size:0.93rem; line-height:1.75; margin:0;">
                이 페이지는 <strong>테스트셋 고객</strong>을 대상으로 합니다.<br><br>
                예측 확률과 실제 이탈 여부를 함께 비교하면
                모델이 실제로 어떤 고객을 고위험으로 판단했는지
                직관적으로 확인할 수 있습니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 전체 테스트셋 ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 전체 테스트셋 예측 비교")

    show_cols = [c for c in ["account_id", "ml_logistic_proba", "dl_mlp_proba", "actual_churn_flag"] if c in pred_df.columns]
    display_df = pred_df[show_cols].copy()
    if "actual_churn_flag" in display_df.columns:
        display_df["actual_churn_flag"] = display_df["actual_churn_flag"].apply(_actual_text)

    st.dataframe(display_df, use_container_width=True)
    _note("테스트셋 전체 고객에 대해 예측 확률과 실제 이탈 여부를 비교한 결과입니다.")
