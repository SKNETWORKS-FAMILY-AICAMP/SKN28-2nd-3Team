# eda_categoricals.py

"""
이 파일은 one-hot encoding된 범주형 변수들을 기준으로
그룹별 고객 수와 이탈률(churn rate)을 요약하고 시각화하는 EDA 모듈이다.

주요 역할:
- 특정 prefix를 가진 dummy 변수 그룹 추출
- 각 범주 그룹별 고객 수와 이탈률 계산
- 범주형 변수별 churn rate 비교 bar plot 생성
- 요약 결과를 CSV와 이미지 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

from pathlib import Path  # 파일 경로 처리용 객체

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리

from src.utils.io import save_csv  # CSV 저장 함수
from src.utils.plot_utils import save_figure  # figure 저장 함수


def summarize_dummy_group(df: pd.DataFrame, prefix: str, target_col: str = "churn_flag") -> pd.DataFrame:
    # 주어진 prefix로 시작하는 dummy 컬럼들만 추출
    dummy_cols = [col for col in df.columns if col.startswith(f"{prefix}_")]

    # 결과를 저장할 리스트
    rows: list[dict] = []

    # dummy 컬럼별로 그룹 요약 수행
    for col in dummy_cols:
        # 해당 그룹(값이 1인 행)만 추출
        subset = df[df[col] == 1].copy()

        # 해당 그룹에 속한 데이터가 없으면 건너뜀
        if subset.empty:
            continue

        # 그룹명, 고객 수, 이탈률 계산 후 저장
        rows.append({
            "group": col.replace(f"{prefix}_", ""),   # prefix 제거 후 그룹명만 남김
            "customer_count": len(subset),            # 해당 그룹 고객 수
            "churn_rate": subset[target_col].mean(),  # 해당 그룹 평균 이탈률
        })

    # 결과가 하나도 없으면 빈 DataFrame 반환
    if not rows:
        return pd.DataFrame(columns=["group", "customer_count", "churn_rate"])

    # DataFrame으로 변환 후 이탈률, 고객 수 기준 내림차순 정렬
    return pd.DataFrame(rows).sort_values(
        ["churn_rate", "customer_count"],
        ascending=[False, False]
    ).reset_index(drop=True)


def plot_dummy_group_summary(summary_df: pd.DataFrame, title: str, output_path: Path) -> None:
    # 요약 데이터가 비어 있으면 시각화하지 않음
    if summary_df.empty:
        return

    # 그래프 크기 설정
    plt.figure(figsize=(10, 6))

    # 그룹별 churn rate bar plot 생성
    plt.bar(summary_df["group"], summary_df["churn_rate"])

    # x축 라벨 회전 (긴 범주명 가독성 확보)
    plt.xticks(rotation=45, ha="right")

    # 축 라벨 및 제목 설정
    plt.ylabel("Churn Rate")
    plt.title(title)

    # 레이아웃 자동 조정
    plt.tight_layout()

    # figure 저장
    save_figure(output_path)

    # 메모리 정리를 위해 figure 닫기
    plt.close()


def run_categorical_eda(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    target_col: str = "churn_flag",
) -> None:
    # 분석할 범주형 변수 prefix와 저장 파일명 매핑
    mapping = {
        "country": "dummy_group_summary_country.csv",
        "industry": "dummy_group_summary_industry.csv",
        "referral_source": "dummy_group_summary_referral.csv",
    }

    # 범주형 변수별 요약 테이블 생성 및 시각화 수행
    for prefix, output_name in mapping.items():
        # 그룹별 고객 수 / 이탈률 요약
        summary = summarize_dummy_group(df, prefix=prefix, target_col=target_col)

        # CSV 저장
        save_csv(summary, tables_dir / output_name)

        # PNG 그래프 저장
        plot_dummy_group_summary(
            summary,
            f"{prefix} 그룹별 Churn Rate",
            plots_dir / output_name.replace('.csv', '.png')
        )