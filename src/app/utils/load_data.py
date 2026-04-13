from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# -----------------------------------
# 경로 설정
# -----------------------------------
try:
    from src.config.paths import (
        PROCESSED_DIR,
        EDA_TABLES_DIR,
        MODELS_OUTPUT_DIR,
        XAI_OUTPUT_DIR,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    EDA_TABLES_DIR = PROJECT_ROOT / "outputs" / "eda" / "tables"
    MODELS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
    XAI_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xai"


# -----------------------------------
# DL 모델 정의
# -----------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------------
# 기본 데이터 로드
# -----------------------------------
@st.cache_data(show_spinner=False)
def load_train_table() -> pd.DataFrame:
    path = PROCESSED_DIR / "train_table_ml.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_analysis_table() -> pd.DataFrame:
    path = PROCESSED_DIR / "analysis_table.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_x_test() -> pd.DataFrame:
    path = PROCESSED_DIR / "X_test.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_y_test() -> pd.DataFrame:
    path = PROCESSED_DIR / "y_test.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# -----------------------------------
# EDA / XAI / 모델 결과 로드
# -----------------------------------
@st.cache_data(show_spinner=False)
def load_group_mean() -> pd.DataFrame:
    path = EDA_TABLES_DIR / "group_mean_by_churn.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    path = MODELS_OUTPUT_DIR / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    path = MODELS_OUTPUT_DIR / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_comparison_tuned() -> pd.DataFrame:
    path = MODELS_OUTPUT_DIR / "model_comparison_tuned.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_threshold_metrics_all_models() -> pd.DataFrame:
    path = MODELS_OUTPUT_DIR / "threshold_metrics_all_models.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_xai_summary() -> pd.DataFrame:
    path = XAI_OUTPUT_DIR / "xai_summary_report.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# -----------------------------------
# 저장 모델 로드
# -----------------------------------
def load_model() -> Any | None:
    path = MODELS_OUTPUT_DIR / "best_model.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_ml_model() -> Any | None:
    return load_model()


def load_random_forest_model() -> Any | None:
    """
    별도 random forest 저장 파일이 있을 때만 사용.
    없으면 None 반환.
    """
    candidate_names = [
        "random_forest_model.pkl",
        "rf_model.pkl",
        "best_random_forest.pkl",
    ]
    for name in candidate_names:
        path = MODELS_OUTPUT_DIR / name
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                return None
    return None


# -----------------------------------
# DL 관련 로드
# -----------------------------------
def _load_dl_feature_columns() -> list[str]:
    path = MODELS_OUTPUT_DIR / "dl_feature_columns.csv"
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
        if "feature" in df.columns:
            return df["feature"].dropna().astype(str).tolist()
        return df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        return []


def _load_dl_scaler() -> Any | None:
    path = MODELS_OUTPUT_DIR / "dl_scaler.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def load_dl_model() -> nn.Module | None:
    model_path = MODELS_OUTPUT_DIR / "dl_model.pth"
    feature_names = _load_dl_feature_columns()

    if not model_path.exists() or not feature_names:
        return None

    try:
        model = MLP(input_dim=len(feature_names))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        return None


# -----------------------------------
# 전처리 공통 함수
# -----------------------------------
def _prepare_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    drop_cols = ["account_id", "customer_id", "id"]
    existing_drop_cols = [c for c in drop_cols if c in X.columns]
    if existing_drop_cols:
        X = X.drop(columns=existing_drop_cols)

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X = X[numeric_cols].copy()

    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    return X


# -----------------------------------
# 단일 행 예측
# -----------------------------------
def predict_ml_row(row: pd.DataFrame) -> float | None:
    model = load_ml_model()
    if model is None:
        return None

    try:
        X = _prepare_numeric_features(row)
        proba = model.predict_proba(X)[0, 1]
        return float(proba)
    except Exception:
        return None


def predict_rf_row(row: pd.DataFrame) -> float | None:
    model = load_random_forest_model()
    if model is None:
        return None

    try:
        X = _prepare_numeric_features(row)
        proba = model.predict_proba(X)[0, 1]
        return float(proba)
    except Exception:
        return None


def predict_dl_row(row: pd.DataFrame) -> float | None:
    dl_model = load_dl_model()
    scaler = _load_dl_scaler()
    feature_names = _load_dl_feature_columns()

    if dl_model is None or scaler is None or not feature_names:
        return None

    try:
        X = _prepare_numeric_features(row)

        missing_cols = [col for col in feature_names if col not in X.columns]
        if missing_cols:
            return None

        X = X[feature_names]
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_proba = dl_model(X_tensor).squeeze().item()

        return float(pred_proba)
    except Exception:
        return None


# -----------------------------------
# 고객별 예측 비교표 생성
# -----------------------------------
@st.cache_data(show_spinner=False)
def build_prediction_comparison() -> pd.DataFrame:
    X_test = load_x_test()
    y_test = load_y_test()

    if X_test.empty:
        return pd.DataFrame()

    result = pd.DataFrame()

    if "account_id" in X_test.columns:
        result["account_id"] = X_test["account_id"].values

    # Logistic / best_model 기반 예측
    ml_model = load_ml_model()
    if ml_model is not None:
        try:
            X_ml = _prepare_numeric_features(X_test)
            result["ml_logistic_proba"] = ml_model.predict_proba(X_ml)[:, 1]
        except Exception:
            result["ml_logistic_proba"] = pd.NA
    else:
        result["ml_logistic_proba"] = pd.NA

    # Random Forest 예측
    rf_model = load_random_forest_model()
    if rf_model is not None:
        try:
            X_rf = _prepare_numeric_features(X_test)
            result["ml_random_forest_proba"] = rf_model.predict_proba(X_rf)[:, 1]
        except Exception:
            result["ml_random_forest_proba"] = pd.NA
    else:
        result["ml_random_forest_proba"] = pd.NA

    # DL 예측
    dl_model = load_dl_model()
    dl_scaler = _load_dl_scaler()
    dl_feature_names = _load_dl_feature_columns()

    if dl_model is not None and dl_scaler is not None and dl_feature_names:
        try:
            X_dl = _prepare_numeric_features(X_test)
            missing_cols = [c for c in dl_feature_names if c not in X_dl.columns]

            if not missing_cols:
                X_dl = X_dl[dl_feature_names]
                X_scaled = dl_scaler.transform(X_dl)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                with torch.no_grad():
                    dl_pred = dl_model(X_tensor).squeeze().numpy()

                result["dl_mlp_proba"] = dl_pred
            else:
                result["dl_mlp_proba"] = pd.NA
        except Exception:
            result["dl_mlp_proba"] = pd.NA
    else:
        result["dl_mlp_proba"] = pd.NA

    # 실제 정답 붙이기
    if not y_test.empty:
        try:
            if "churn_flag" in y_test.columns:
                result["actual_churn_flag"] = y_test["churn_flag"].values
            else:
                result["actual_churn_flag"] = y_test.iloc[:, 0].values
        except Exception:
            result["actual_churn_flag"] = pd.NA

    return result


# -----------------------------------
# threshold map
# -----------------------------------
def get_tuned_threshold_map() -> dict[str, float]:
    tuned_df = load_model_comparison_tuned()

    if tuned_df.empty:
        return {}

    required_cols = {"model", "threshold"}
    if not required_cols.issubset(set(tuned_df.columns)):
        return {}

    mapping: dict[str, float] = {}
    for _, row in tuned_df.iterrows():
        try:
            mapping[str(row["model"])] = float(row["threshold"])
        except Exception:
            continue
    return mapping