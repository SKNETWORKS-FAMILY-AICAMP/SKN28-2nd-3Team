# eda_by_churn.py

"""
이 파일은 churn 여부(이탈 vs 비이탈)에 따라 주요 feature들의 차이를 탐색하기 위한
EDA(탐색적 데이터 분석) 모듈이다.

주요 역할:
- 수치형 변수 중 타겟 변수(churn_flag)를 제외한 컬럼 추출
- churn / non-churn 그룹 간 평균값 비교 테이블 생성
- 주요 feature에 대해 그룹별 평균 비교 시각화(bar plot) 생성
- 분석 결과를 CSV 및 이미지 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

from pathlib import Path  # 파일 경로 객체 처리

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리

from src.utils.io import save_csv  # CSV 저장 함수
from src.utils.plot_utils import save_figure  # 이미지 저장 함수


def _safe_numeric_columns(df: pd.DataFrame, target_col: str = "churn_flag") -> list[str]:
    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # 타겟 변수는 제외
    return [col for col in numeric_cols if col != target_col]


def _make_group_mean_table(df: pd.DataFrame, target_col: str = "churn_flag") -> pd.DataFrame:
    rows = []  # 결과 저장 리스트

    # 수치형 컬럼을 하나씩 순회하며 평균 계산
    for col in _safe_numeric_columns(df, target_col=target_col):

        # 비이탈 그룹 평균
        non_churn_mean = df.loc[df[target_col] == 0, col].mean()

        # 이탈 그룹 평균
        churn_mean = df.loc[df[target_col] == 1, col].mean()

        # 결과를 딕셔너리 형태로 저장
        rows.append({
            "feature": col,
            "non_churn_mean": non_churn_mean,
            "churn_mean": churn_mean,
            "diff_churn_minus_nonchurn": churn_mean - non_churn_mean,  # 두 그룹 간 차이
        })

    # 리스트를 DataFrame으로 변환
    result = pd.DataFrame(rows)

    # 결과가 비어있지 않다면, 차이가 큰 순으로 정렬
    if not result.empty:
        result = result.sort_values("diff_churn_minus_nonchurn", ascending=False).reset_index(drop=True)

    return result  # 그룹 평균 비교 테이블 반환


def _plot_mean_comparison(df: pd.DataFrame, feature: str, output_path: Path, target_col: str = "churn_flag") -> None:
    # churn 여부별 평균값 계산 (index 라벨 변경)
    group_means = df.groupby(target_col, dropna=False)[feature].mean().rename(index={0: "Non-Churn", 1: "Churn"})

    # 그래프 크기 설정
    plt.figure(figsize=(8, 5))

    # bar plot 생성
    plt.bar(group_means.index.astype(str), group_means.values)

    # 그래프 제목 및 축 라벨 설정
    plt.title(f"{feature} 평균 비교")
    plt.ylabel(feature)
    plt.xlabel("Churn Group")

    # 레이아웃 정리
    plt.tight_layout()

    # 이미지 파일로 저장
    save_figure(output_path)

    # 메모리 정리를 위해 figure 닫기
    plt.close()


def run_churn_comparison_eda(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    target_col: str = "churn_flag",
) -> None:

    # 타겟 컬럼이 존재하지 않으면 에러 발생
    if target_col not in df.columns:
        raise KeyError(f"'{target_col}' 컬럼이 데이터프레임에 없습니다.")

    # churn vs non-churn 평균 비교 테이블 생성
    group_mean_df = _make_group_mean_table(df, target_col=target_col)

    # 결과 CSV 저장
    save_csv(group_mean_df, tables_dir / "group_mean_by_churn.csv")

    # 시각화 대상 주요 feature 목록 정의
    preferred = [
        ("total_usage_count", "bar_mean_by_churn_usage.png"),
        ("error_rate", "bar_mean_by_churn_error_rate.png"),
        ("avg_satisfaction_score", "bar_mean_by_churn_satisfaction.png"),
        ("health_score", "bar_mean_by_churn_health_score.png"),
    ]

    # feature별로 그래프 생성
    for feature, filename in preferred:

        # 해당 feature가 데이터에 존재할 경우만 실행
        if feature in df.columns:
            _plot_mean_comparison(df, feature, plots_dir / filename, target_col=target_col)