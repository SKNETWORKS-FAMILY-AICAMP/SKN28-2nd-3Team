# eda_numeric.py

"""
이 파일은 수치형 변수들의 분포, 왜도, 상관관계 등을 점검하여
모델링 전에 데이터의 전반적인 특성을 탐색하는 EDA 모듈이다.

주요 역할:
- 수치형 변수 목록 추출
- 수치형 변수의 기초통계 요약 생성
- 변수별 왜도(skewness) 계산
- 주요 수치형 변수 간 상관관계 행렬 생성 및 heatmap 시각화
- 각 수치형 변수와 churn_flag 간 상관관계 계산
- 주요 변수들의 분포(histogram) 시각화
- 분석 결과를 CSV 및 이미지 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

from pathlib import Path  # 파일 경로 처리용 객체

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리

from src.config.settings import KEY_NUMERIC_FEATURES  # 주요 수치형 변수 목록
from src.utils.io import save_csv  # CSV 저장 함수
from src.utils.plot_utils import save_figure  # figure 저장 함수


def _numeric_columns(df: pd.DataFrame, target_col: str = "churn_flag") -> list[str]:
    # 데이터프레임에서 수치형 컬럼만 추출
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # 타겟 컬럼은 제외
    return [c for c in numeric_cols if c not in {target_col}]


def run_numeric_eda(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    target_col: str = "churn_flag",
) -> None:
    # -----------------------------------
    # 1. 수치형 컬럼 목록 추출
    # -----------------------------------
    numeric_cols = _numeric_columns(df, target_col)

    # -----------------------------------
    # 2. 기초통계 요약 생성
    # -----------------------------------
    numeric_summary = (
        df[numeric_cols]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    # 수치형 변수 요약 통계 저장
    save_csv(numeric_summary, tables_dir / "numeric_summary.csv")

    # -----------------------------------
    # 3. 왜도(skewness) 계산
    # -----------------------------------
    skewness = (
        df[numeric_cols]
        .skew(numeric_only=True)
        .rename("skewness")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    # 왜도 결과 저장
    save_csv(skewness, tables_dir / "skewness_summary.csv")

    # -----------------------------------
    # 4. 주요 변수 간 상관관계 분석
    # -----------------------------------
    # 설정 파일에 정의된 주요 수치형 변수 중 실제 존재하는 컬럼만 선택
    corr_cols = [c for c in KEY_NUMERIC_FEATURES if c in df.columns]

    if corr_cols:
        # 상관관계 행렬 계산
        corr = df[corr_cols].corr(numeric_only=True)

        # 상관관계 행렬 CSV 저장
        save_csv(
            corr.reset_index().rename(columns={"index": "feature"}),
            tables_dir / "correlation_matrix_key_features.csv"
        )

        # heatmap 시각화
        plt.figure(figsize=(12, 8))
        plt.imshow(corr, aspect="auto")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Heatmap (Key Features)")
        plt.colorbar()
        plt.tight_layout()
        save_figure(plots_dir / "correlation_heatmap_key_features.png")
        plt.close()

    # -----------------------------------
    # 5. 타겟 변수와의 상관관계 분석
    # -----------------------------------
    if target_col in df.columns:
        corr_target = (
            df[numeric_cols + [target_col]]
            .corr(numeric_only=True)[target_col]
            .drop(target_col)
            .sort_values(ascending=False)
            .rename("correlation_with_churn")
            .reset_index()
            .rename(columns={"index": "feature"})
        )

        # churn과의 상관관계 저장
        save_csv(corr_target, tables_dir / "correlation_with_churn.csv")

        # 사전 중요도 점검용 파일로도 저장
        save_csv(
            corr_target.rename(columns={"correlation_with_churn": "score"}),
            tables_dir / "feature_importance_precheck.csv"
        )

    # -----------------------------------
    # 6. 주요 수치형 변수 분포 시각화
    # -----------------------------------
    hist_targets = [
        ("account_age_days", "hist_account_age_days.png"),
        ("total_subscriptions", "hist_total_subscriptions.png"),
        ("avg_mrr_amount", "hist_avg_mrr_amount.png"),
        ("days_since_last_usage", "hist_days_since_last_usage.png"),
        ("health_score", "hist_health_score.png"),
    ]

    # 지정한 주요 변수들에 대해 histogram 생성
    for feature, filename in hist_targets:
        if feature not in df.columns:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(df[feature].dropna(), bins=30)
        plt.title(f"{feature} Distribution")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        save_figure(plots_dir / filename)
        plt.close()