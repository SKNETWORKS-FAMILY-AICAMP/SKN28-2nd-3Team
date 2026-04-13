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


def _format_probability(prob: float | None) -> str:
    if prob is None or pd.isna(prob):
        return "N/A"
    return f"{prob:.3f}"


def _risk_label(prob: float | None, threshold: float | None) -> str:
    if prob is None or pd.isna(prob) or threshold is None:
        return "판단 불가"
    return "고위험" if prob >= threshold else "안정"


def _find_best_model_name(
    base_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
) -> str | None:
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


def _action_guide(prob: float | None, threshold: float | None) -> str:
    if prob is None or pd.isna(prob) or threshold is None:
        return "예측 결과가 충분하지 않아 후속 액션을 제안하기 어렵다."

    if prob >= threshold + 0.15:
        return "즉시 케어 대상이다. 전담 CS 연결, 사용 저해 요인 확인, 핵심 기능 재안내가 필요하다."

    if prob >= threshold:
        return "주의 고객이다. 리텐션 메시지 발송, 사용량 추적, 이슈 여부 점검이 적절하다."

    return "현재는 안정 고객으로 볼 수 있으나, 사용량 감소나 오류율 상승 여부를 지속 모니터링하는 것이 좋다."


def _actual_flag_to_text(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    try:
        value = int(value)
    except Exception:
        return "N/A"

    if value == 1:
        return "1 (이탈)"
    if value == 0:
        return "0 (유지)"
    return str(value)


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
    st.markdown("## 고객별 예측 및 액션 제안")

    pred_df = build_prediction_comparison()
    X_test = load_x_test()
    base_df = load_model_comparison()
    tuned_df = load_model_comparison_tuned()
    threshold_map = get_tuned_threshold_map()

    if pred_df.empty or X_test.empty:
        st.warning("예측 비교용 데이터가 없습니다.")
        return

    st.markdown(
        f"""
        <div class="section-card">
            <h4 style="margin-top:0;">목적</h4>
            <p style="line-height:1.7; margin-bottom:0;">
            개별 고객의 이탈 위험을 확인하고, 실제 이탈 여부와 비교하는 단계이다.<br>
            모델 예측이 실제 고객 행동과 어떻게 연결되는지 직관적으로 확인할 수 있다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "본 프로젝트에서 **churn_flag**는 고객 이탈 여부를 의미한다. "
        "**1은 이탈 고객**, **0은 유지 고객**으로 해석한다."
    )

    st.caption(
        "Random Forest는 본 프로젝트에서 성능 비교용 모델로 활용하였고, "
        "고객별 예측 화면에서는 운영 및 해석 중심의 Logistic Regression, DL MLP 결과만 제시하였다."
    )

    account_col = "account_id" if "account_id" in pred_df.columns else None
    if account_col is None:
        st.error("account_id 컬럼이 없어 고객별 비교가 불가능합니다.")
        return

    account_list = pred_df[account_col].dropna().astype(str).tolist()
    if not account_list:
        st.warning("선택 가능한 account_id가 없습니다.")
        return

    selected_account = st.selectbox("분석할 고객(account_id) 선택", account_list)

    row_pred = pred_df[pred_df[account_col].astype(str) == selected_account].head(1)

    if account_col in X_test.columns:
        row_x = X_test[X_test[account_col].astype(str) == selected_account].head(1)
    else:
        row_x = pd.DataFrame()

    if row_pred.empty:
        st.warning("선택한 고객의 예측 결과를 찾을 수 없습니다.")
        return

    lr_prob = row_pred["ml_logistic_proba"].iloc[0] if "ml_logistic_proba" in row_pred.columns else pd.NA
    dl_prob = row_pred["dl_mlp_proba"].iloc[0] if "dl_mlp_proba" in row_pred.columns else pd.NA
    actual_flag = row_pred["actual_churn_flag"].iloc[0] if "actual_churn_flag" in row_pred.columns else pd.NA

    lr_th = threshold_map.get("logistic_regression", 0.5)
    dl_th = threshold_map.get("DL_MLP", 0.5)

    col1, col2, col3 = st.columns([1, 1, 0.9])

    with col1:
        st.metric(
            "Logistic Regression",
            _format_probability(lr_prob),
            help="선형 기준 churn 확률",
        )
        st.caption(f"Threshold: {lr_th:.2f} · {_risk_label(lr_prob, lr_th)}")

    with col2:
        st.metric(
            "DL MLP",
            _format_probability(dl_prob),
            help="복합 상호작용 반영 churn 확률",
        )
        st.caption(f"Threshold: {dl_th:.2f} · {_risk_label(dl_prob, dl_th)}")

    with col3:
        st.metric("실제 churn_flag", _actual_flag_to_text(actual_flag))
        st.caption("1 = 이탈, 0 = 유지")

    if all(pd.isna(p) for p in [lr_prob, dl_prob]):
        st.warning(
            "현재 선택 고객에 대해 예측 확률을 불러오지 못해 N/A로 표시되고 있다. "
            "이는 저장 모델 파일이 없거나, 예측 단계에서 입력 컬럼 구성이 맞지 않을 때 발생할 수 있다."
        )

    st.markdown("### 추천 액션")

    best_model_name = _find_best_model_name(base_df, tuned_df)
    best_prob = None
    best_th = None

    if best_model_name == "logistic_regression":
        best_prob, best_th = lr_prob, lr_th
    elif best_model_name == "DL_MLP":
        best_prob, best_th = dl_prob, dl_th
    else:
        if not pd.isna(lr_prob):
            best_model_name = "logistic_regression"
            best_prob, best_th = lr_prob, lr_th
        elif not pd.isna(dl_prob):
            best_model_name = "DL_MLP"
            best_prob, best_th = dl_prob, dl_th

    if best_model_name is None:
        best_model_name = "N/A"

    st.info(
        f"""
현재 고객별 예측 화면에서는 **{best_model_name}** 기준으로 해석하였다.  

**{_action_guide(best_prob, best_th)}**
"""
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("선택 고객 feature 미리보기")
        if not row_x.empty:
            display_cols = [c for c in row_x.columns if c != "account_id"][:15]
            preview_cols = [account_col] + display_cols if account_col in row_x.columns else display_cols
            preview_df = row_x[preview_cols].copy()
            st.dataframe(preview_df, use_container_width=True)
            _center_note(
                "이 표는 선택한 고객의 주요 feature 값을 요약해서 보여준다. "
                "예측 결과를 단순 수치로만 보는 것이 아니라, 어떤 입력 특성을 가진 고객인지 함께 해석하기 위한 보조 자료라고 볼 수 있다."
            )
        else:
            st.caption("원본 feature 정보를 불러오지 못했습니다.")

    with right:
        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin-top:0;">해석 포인트</h4>
                <p style="line-height:1.8; margin-bottom:0;">
                이 페이지는 전체 고객이 아니라 <b>테스트셋 고객</b>을 대상으로 한다.
                예측 확률과 실제 이탈 여부를 함께 비교함으로써,
                모델이 실제로 어떤 고객을 고위험으로 판단했는지 직관적으로 확인할 수 있다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("전체 테스트셋 고객 예측 비교표")

    show_cols = [
        c
        for c in [
            "account_id",
            "ml_logistic_proba",
            "dl_mlp_proba",
            "actual_churn_flag",
        ]
        if c in pred_df.columns
    ]

    display_df = pred_df[show_cols].copy()

    if "actual_churn_flag" in display_df.columns:
        display_df["actual_churn_flag"] = display_df["actual_churn_flag"].apply(_actual_flag_to_text)

    st.dataframe(display_df, use_container_width=True)
    _center_note(
        "이 표는 테스트셋 전체 고객에 대해 예측 확률과 실제 이탈 여부를 함께 비교한 결과이다. "
        "즉, 개별 사례를 넘어서 모델이 전반적으로 어떤 고객을 고위험으로 판단했는지 확인하는 요약 표로 해석할 수 있다."
    )