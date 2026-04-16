# load_data.py

from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pickle  # pickle 기반 객체 로드용
from pathlib import Path  # 파일 경로 처리용 객체
from typing import Any  # 다양한 타입 허용

import joblib  # sklearn 모델 로딩용
import pandas as pd  # 데이터프레임 처리
import streamlit as st  # Streamlit 캐싱 및 UI
import torch  # 딥러닝 모델 처리
import torch.nn as nn  # 신경망 정의


# -----------------------------------
# 경로 설정
# -----------------------------------
try:
    # 정상적인 프로젝트 구조에서 경로 import
    from src.config.paths import (
        PROCESSED_DIR,       # 전처리 데이터 경로
        EDA_TABLES_DIR,      # EDA 결과 경로
        MODELS_OUTPUT_DIR,   # 모델 결과 경로
        XAI_OUTPUT_DIR,      # XAI 결과 경로
    )
except ImportError:
    # Streamlit Cloud 등에서 import 실패 시 fallback 경로 설정
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
        super().__init__()  # 부모 클래스 초기화
        # 다층 퍼셉트론 구조 정의
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # 입력 → 64 노드
            nn.ReLU(),                 # 활성화 함수
            nn.Dropout(0.3),           # 과적합 방지
            nn.Linear(64, 32),         # 64 → 32
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),          # 출력층
            nn.Sigmoid(),              # 확률 값으로 변환
        )

    def forward(self, x):
        return self.model(x)  # 순전파 정의


# -----------------------------------
# 기본 데이터 로드
# -----------------------------------
@st.cache_data(show_spinner=False)  # 캐싱으로 성능 최적화
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
        return joblib.load(path)  # sklearn 모델 로드
    except Exception:
        return None


def load_ml_model() -> Any | None:
    return load_model()  # 기본 모델 alias


def load_random_forest_model() -> Any | None:
    # 여러 파일명을 순회하며 RF 모델 탐색
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
        model = MLP(input_dim=len(feature_names))  # 모델 구조 생성
        state_dict = torch.load(model_path, map_location="cpu")  # weight 로드
        model.load_state_dict(state_dict)  # weight 적용
        model.eval()  # 추론 모드 설정
        return model
    except Exception:
        return None


# -----------------------------------
# 전처리 공통 함수
# -----------------------------------
def _prepare_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()  # 원본 보호

    # ID 컬럼 제거
    drop_cols = ["account_id", "customer_id", "id"]
    existing_drop_cols = [c for c in drop_cols if c in X.columns]
    if existing_drop_cols:
        X = X.drop(columns=existing_drop_cols)

    # bool → int 변환
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # 숫자형 컬럼만 선택
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X = X[numeric_cols].copy()

    # 결측치 median으로 대체
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

        # feature mismatch 체크
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