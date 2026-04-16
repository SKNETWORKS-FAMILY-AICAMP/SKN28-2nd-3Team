# eda_missingness.py

"""
이 파일은 데이터셋의 결측치 분포와 결측 여부에 따른 churn 차이를 탐색하기 위한
EDA(탐색적 데이터 분석) 모듈이다.

주요 역할:
- 컬럼별 결측치 개수 및 결측 비율 계산
- 결측 여부(missing_flag)에 따른 고객 수와 churn rate 요약
- 결측 패턴을 heatmap으로 시각화
- 결측치 관련 분석 결과를 CSV 및 이미지 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

from pathlib import Path  # 파일 경로 처리용 객체

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리

from src.utils.io import save_csv  # CSV 저장 함수
from src.utils.plot_utils import save_figure  # figure 저장 함수


def run_missingness_eda(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    target_col: str = "churn_flag",
) -> None:
    # 결과 저장 폴더가 없으면 생성
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # 1. 컬럼별 결측치 개수 계산
    # -----------------------------------
    missing_counts = df.isna().sum().sort_values(ascending=False).rename("missing_count").reset_index()
    missing_counts.columns = ["column", "missing_count"]

    # 결측치 개수 테이블 저장
    save_csv(missing_counts, tables_dir / "missing_counts.csv")

    # -----------------------------------
    # 2. 컬럼별 결측 비율 계산
    # -----------------------------------
    missing_ratio = (df.isna().mean() * 100).round(2).rename("missing_ratio_pct").reset_index()
    missing_ratio.columns = ["column", "missing_ratio_pct"]

    # 결측 비율 테이블 저장
    save_csv(missing_ratio, tables_dir / "missing_ratio.csv")

    # -----------------------------------
    # 3. 타겟 컬럼 존재 여부 확인
    # -----------------------------------
    if target_col not in df.columns:
        raise KeyError(f"'{target_col}' 컬럼이 데이터프레임에 없습니다.")

    # -----------------------------------
    # 4. 결측 여부와 churn 관계 요약
    # -----------------------------------
    records: list[pd.DataFrame] = []

    # 각 컬럼별로 missing 여부와 churn 관계를 분석
    for col in df.columns:
        # 타겟 컬럼 자체는 제외
        if col == target_col:
            continue

        # 현재 컬럼의 결측 여부를 0/1로 변환하여 임시 데이터프레임 생성
        tmp = pd.DataFrame({
            "column": col,
            "missing_flag": df[col].isna().astype(int),  # 결측이면 1, 아니면 0
            target_col: df[target_col],
        })

        # 컬럼별 / 결측 여부별 고객 수와 churn rate 집계
        summary = (
            tmp.groupby(["column", "missing_flag"], dropna=False)[target_col]
            .agg(customer_count="count", churn_rate="mean")
            .reset_index()
        )

        # 결과를 리스트에 추가
        records.append(summary)

    # 전체 컬럼 결과를 하나로 결합
    missing_vs_churn = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["column", "missing_flag", "customer_count", "churn_rate"]
    )

    # 결측 여부와 churn 관계 테이블 저장
    save_csv(missing_vs_churn, tables_dir / "missing_vs_churn.csv")

    # -----------------------------------
    # 5. 결측 패턴 heatmap 생성
    # -----------------------------------
    # 결측 여부를 0/1 행렬로 변환
    heatmap_df = df.isna().astype(int)

    # 컬럼이 너무 많으면 앞 60개까지만 시각화
    if heatmap_df.shape[1] > 60:
        heatmap_df = heatmap_df.iloc[:, :60]

    # 그래프 크기 설정
    plt.figure(figsize=(12, 6))

    # 결측 패턴 heatmap 시각화
    plt.imshow(heatmap_df.T, aspect="auto")

    # y축에 컬럼명 표시
    plt.yticks(range(len(heatmap_df.columns)), heatmap_df.columns)

    # x축 눈금은 생략
    plt.xticks([])

    # 그래프 제목 설정
    plt.title("Missing Pattern Heatmap")

    # 레이아웃 자동 정리
    plt.tight_layout()

    # figure 저장
    save_figure(plots_dir / "missing_pattern_heatmap.png")

    # 메모리 정리를 위해 figure 닫기
    plt.close()