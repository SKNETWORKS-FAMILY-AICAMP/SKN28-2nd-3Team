# eda_visualization.py

"""
이 파일은 EDA 과정에서 자주 사용하는 기본 시각화 그래프를 생성하고 저장하는 모듈이다.

주요 역할:
- 수치형 변수의 분포(histogram) 저장
- churn 여부에 따른 평균 비교 bar plot 저장
- 타겟 변수 분포 bar plot 저장
- 주요 변수들의 상관관계 heatmap 저장
- 결측 패턴 heatmap 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리

from src.utils.plot_utils import apply_plot_style  # 공통 plot 스타일 적용 함수


def save_histograms(df: pd.DataFrame, features: list[str], output_dir) -> None:
    # 공통 그래프 스타일 적용
    apply_plot_style()

    # 지정한 feature마다 histogram 생성
    for col in features:
        plt.figure()

        # 결측치를 제외한 뒤 histogram 생성
        df[col].dropna().hist(bins=30)

        # 제목 및 축 라벨 설정
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("count")

        # 레이아웃 자동 정리
        plt.tight_layout()

        # 이미지 파일로 저장
        plt.savefig(output_dir / f"hist_{col}.png")

        # 메모리 정리를 위해 figure 닫기
        plt.close()


def save_bar_means(df: pd.DataFrame, features: list[str], target_col: str, output_dir) -> None:
    # 공통 그래프 스타일 적용
    apply_plot_style()

    # 타겟 변수 기준으로 feature 평균 계산
    grouped = df.groupby(target_col)[features].mean()

    # feature별 평균 비교 bar plot 생성
    for col in features:
        plt.figure()

        # 그룹별 평균 bar plot
        grouped[col].plot(kind="bar")

        # 제목 및 축 라벨 설정
        plt.title(f"mean {col} by churn")
        plt.xlabel(target_col)
        plt.ylabel("mean")

        # 레이아웃 자동 정리
        plt.tight_layout()

        # 이미지 파일로 저장
        plt.savefig(output_dir / f"bar_mean_by_churn_{col}.png")

        # 메모리 정리를 위해 figure 닫기
        plt.close()


def save_target_distribution(df: pd.DataFrame, target_col: str, output_dir) -> None:
    # 공통 그래프 스타일 적용
    apply_plot_style()

    plt.figure()

    # 타겟 변수 값 분포를 bar plot으로 시각화
    df[target_col].value_counts().sort_index().plot(kind="bar")

    # 제목 및 축 라벨 설정
    plt.title("target distribution overall")
    plt.xlabel(target_col)
    plt.ylabel("count")

    # 레이아웃 자동 정리
    plt.tight_layout()

    # 이미지 파일 저장
    plt.savefig(output_dir / "target_distribution_overall.png")

    # 메모리 정리를 위해 figure 닫기
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, features: list[str], target_col: str, output_dir) -> None:
    import numpy as np  # 수치 계산 라이브러리 (현재 코드에서는 직접 사용되지는 않지만 import 유지)

    # 공통 그래프 스타일 적용
    apply_plot_style()

    # -----------------------------------
    # 1. 상관관계 heatmap 생성
    # -----------------------------------
    # 지정한 feature와 target_col을 포함한 상관관계 행렬 계산
    corr = df[features + [target_col]].corr(numeric_only=True)

    plt.figure(figsize=(12, 10))

    # 상관관계 행렬 시각화
    plt.imshow(corr, aspect="auto")

    # 축 눈금 및 컬럼명 표시
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)

    # color bar 추가
    plt.colorbar()

    # 레이아웃 자동 정리
    plt.tight_layout()

    # heatmap 이미지 저장
    plt.savefig(output_dir / "correlation_heatmap_key_features.png")

    # 메모리 정리를 위해 figure 닫기
    plt.close()

    # -----------------------------------
    # 2. 결측 패턴 heatmap 생성
    # -----------------------------------
    # feature들의 결측 여부를 0/1 행렬로 변환
    missing_matrix = df[features].isna().astype(int)

    plt.figure(figsize=(12, 6))

    # 결측 패턴 시각화
    plt.imshow(missing_matrix.T, aspect="auto")

    # y축에 feature 이름 표시
    plt.yticks(range(len(features)), features)

    # 레이아웃 자동 정리
    plt.tight_layout()

    # heatmap 이미지 저장
    plt.savefig(output_dir / "missing_pattern_heatmap.png")

    # 메모리 정리를 위해 figure 닫기
    plt.close()