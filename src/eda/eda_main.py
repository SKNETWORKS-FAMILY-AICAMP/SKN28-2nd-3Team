# eda_main.py

"""
이 파일은 전처리 및 feature engineering이 완료된 학습용 데이터를 불러와
EDA(탐색적 데이터 분석)를 일괄 실행하는 메인 실행 모듈이다.

주요 역할:
- EDA 결과 저장 폴더 생성
- 분석 대상 데이터(train_table_ml.csv) 로드
- 타겟 변수(churn_flag) 분포 요약 및 시각화
- 결측치, 수치형 변수, 범주형 변수, churn 그룹 비교 EDA 순차 실행
- 전체 EDA 결과 요약 리포트 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리

# 경로 설정 (processed 데이터 및 EDA 산출물 저장 경로)
from src.config.paths import PROCESSED_DIR, EDA_TABLES_DIR, EDA_PLOTS_DIR

# 세부 EDA 모듈 import
from src.eda.eda_missingness import run_missingness_eda  # 결측치 분석
from src.eda.eda_numeric import run_numeric_eda  # 수치형 변수 분석
from src.eda.eda_categoricals import run_categorical_eda  # 범주형 변수 분석
from src.eda.eda_by_churn import run_churn_comparison_eda  # churn 여부별 비교 분석

# 공통 plot 스타일 적용 함수
from src.utils.plot_utils import apply_plot_style


def ensure_output_dirs() -> None:
    # EDA 결과를 저장할 tables / plots 폴더 생성
    # 이미 존재하면 에러 없이 통과
    EDA_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data() -> pd.DataFrame:
    # 분석 대상 파일 경로 지정
    file_path = PROCESSED_DIR / "train_table_ml.csv"

    # 파일이 없으면 명확한 에러 메시지 출력
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {file_path}\n"
            "먼저 make_train_table.py를 실행해서 train_table_ml.csv를 생성하세요."
        )

    # CSV 파일 로드 후 반환
    return pd.read_csv(file_path)


def build_target_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    # 전체 데이터 기준 타겟 분포 요약 생성
    rows = [{
        "dataset": "overall",                        # 전체 데이터셋 기준
        "row_count": len(df),                       # 전체 행 수
        "churn_count": int(df["churn_flag"].sum()), # churn 고객 수
        "churn_rate": float(df["churn_flag"].mean()),  # churn 비율
    }]

    # DataFrame 형태로 반환
    return pd.DataFrame(rows)


def main() -> None:
    # 공통 plot 스타일 적용
    apply_plot_style()

    # 결과 저장 폴더 생성
    ensure_output_dirs()

    # 분석 대상 데이터 로드
    df = load_processed_data()

    # -----------------------------------
    # 1. 타겟 분포 요약 저장
    # -----------------------------------
    target_summary = build_target_distribution_summary(df)
    target_summary.to_csv(
        EDA_TABLES_DIR / "target_distribution_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------
    # 2. 타겟 분포 시각화
    # -----------------------------------
    plt.figure(figsize=(6, 4))

    # churn_flag 값별 빈도 계산
    counts = df["churn_flag"].value_counts().sort_index()

    # bar plot 생성
    plt.bar(["Non-Churn", "Churn"], counts.values)

    # 제목 설정
    plt.title("Target Distribution")

    # 레이아웃 정리
    plt.tight_layout()

    # 그래프 저장
    plt.savefig(EDA_PLOTS_DIR / "target_distribution_overall.png", bbox_inches="tight")

    # figure 종료
    plt.close()

    # -----------------------------------
    # 3. 세부 EDA 모듈 순차 실행
    # -----------------------------------
    print("[1/4] Missingness EDA 실행 중...")
    run_missingness_eda(df, EDA_TABLES_DIR, EDA_PLOTS_DIR)

    print("[2/4] Numeric EDA 실행 중...")
    run_numeric_eda(df, EDA_TABLES_DIR, EDA_PLOTS_DIR)

    print("[3/4] Categorical EDA 실행 중...")
    run_categorical_eda(df, EDA_TABLES_DIR, EDA_PLOTS_DIR)

    print("[4/4] Churn comparison EDA 실행 중...")
    run_churn_comparison_eda(df, EDA_TABLES_DIR, EDA_PLOTS_DIR)

    # -----------------------------------
    # 4. 전체 EDA 요약 리포트 저장
    # -----------------------------------
    summary = pd.DataFrame([{
        "row_count": len(df),                       # 전체 데이터 수
        "column_count": df.shape[1],               # 전체 컬럼 수
        "churn_rate": float(df["churn_flag"].mean()),  # 전체 churn 비율
    }])

    summary.to_csv(
        EDA_TABLES_DIR / "eda_summary_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 실행 완료 메시지 출력
    print("EDA 완료")
    print(f"- tables: {EDA_TABLES_DIR}")
    print(f"- plots : {EDA_PLOTS_DIR}")


# 스크립트 직접 실행 시 main 함수 호출
if __name__ == "__main__":
    main()