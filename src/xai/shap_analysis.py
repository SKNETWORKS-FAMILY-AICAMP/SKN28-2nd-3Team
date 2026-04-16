# shap_analysis.py

"""
이 파일은 학습된 트리 기반 모델에 대해 SHAP 분석을 수행하여,
전역 설명(global explanation), 로컬 설명(local explanation), 이탈 사유 매핑 리포트,
그리고 주요 SHAP 시각화 결과를 저장하는 XAI 실행 모듈이다.

주요 역할:
- 저장된 best model 불러오기
- 학습 데이터와 분석용 테이블 로드
- SHAP 분석 대상 입력 데이터 구성
- SHAP value 계산 및 전역 중요도 요약 저장
- 샘플 단위 로컬 설명 결과 저장
- 실제 이탈 사유 코드와 연결한 reason mapping 리포트 생성
- SHAP summary, bar, dependence plot 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

import warnings  # SHAP 실행 시 발생할 수 있는 불필요한 경고 제어용 모듈

import joblib  # 저장된 모델 객체 로드용 라이브러리
import matplotlib.pyplot as plt  # 시각화 결과 저장용 라이브러리
import pandas as pd  # 데이터프레임 처리 라이브러리
import shap  # SHAP 설명 가능성 분석 라이브러리

from src.config.paths import PROCESSED_DIR, MODELS_OUTPUT_DIR, XAI_OUTPUT_DIR  # 데이터/모델/XAI 산출물 경로
from src.config.settings import TARGET_COL  # 타겟 변수명 설정값
from src.xai.global_explanations import save_global_shap_summary  # 전역 SHAP 요약 저장 함수
from src.xai.local_explanations import save_local_explanations  # 로컬 SHAP 설명 저장 함수
from src.xai.reason_mapping import build_reason_mapping_report  # 이탈 사유 매핑 리포트 생성 함수
from src.utils.io import read_csv  # CSV 읽기 함수
from src.utils.plot_utils import apply_plot_style  # 공통 plot 스타일 적용 함수
from src.utils.logger import get_logger  # 로거 생성 함수

# 현재 파일 기준 logger 생성
logger = get_logger(__name__)


def main() -> None:
    # -----------------------------------
    # 1. 모델 및 데이터 로드
    # -----------------------------------
    # 학습이 완료된 최종 트리 기반 모델 로드
    model = joblib.load(MODELS_OUTPUT_DIR / "best_model.pkl")

    # SHAP 분석에 사용할 train 데이터 로드
    train = read_csv(PROCESSED_DIR / "train.csv")

    # 실제 이탈 사유 코드가 포함된 분석용 테이블 로드
    analysis = read_csv(PROCESSED_DIR / "analysis_table.csv")

    # -----------------------------------
    # 2. SHAP 입력 데이터 구성
    # -----------------------------------
    # 타겟 변수와 식별자(account_id)는 설명 대상 feature에서 제외
    X = train.drop(columns=[TARGET_COL, "account_id"], errors="ignore")

    # SHAP 계산량을 줄이기 위해 샘플 수가 너무 많으면 200개만 표본 추출
    if len(X) > 200:
        X = X.sample(200, random_state=42)

    # -----------------------------------
    # 3. 공통 시각화 스타일 적용
    # -----------------------------------
    apply_plot_style()

    # -----------------------------------
    # 4. SHAP value 계산
    # -----------------------------------
    # SHAP 라이브러리 사용 시 발생할 수 있는 일부 경고를 숨김
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 트리 기반 모델용 SHAP explainer 생성
        explainer = shap.TreeExplainer(model)

        # 입력 데이터 X에 대한 shap values 계산
        shap_values = explainer.shap_values(X)

    # -----------------------------------
    # 5. SHAP 결과 형태 정리
    # -----------------------------------
    # 이진 분류 모델에서는 shap_values가 list 형태로 반환될 수 있음
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]  # 양성 클래스(보통 churn=1)에 해당하는 shap 값 사용
    else:
        shap_matrix = shap_values

    # 일부 버전/모델에서는 3차원 형태로 반환될 수 있어 차원 정리
    if getattr(shap_matrix, "ndim", 0) == 3:
        shap_matrix = shap_matrix[:, :, 1]

    # -----------------------------------
    # 6. 전역 SHAP 요약 저장
    # -----------------------------------
    # feature별 평균 절대 SHAP 값 계산
    mean_abs = abs(shap_matrix).mean(axis=0)

    # 전역 중요도 요약 테이블 저장
    global_summary = save_global_shap_summary(
        mean_abs,
        X.columns,
        XAI_OUTPUT_DIR / "xai_summary_report.csv"
    )

    # -----------------------------------
    # 7. 로컬 설명 저장
    # -----------------------------------
    # 개별 샘플 단위 feature 기여도를 CSV로 저장
    save_local_explanations(
        shap_matrix,
        X.reset_index(drop=True),
        XAI_OUTPUT_DIR / "local_explanation_sample_0.csv"
    )

    # -----------------------------------
    # 8. reason mapping 리포트 생성
    # -----------------------------------
    # 모델이 중요하게 본 feature와 실제 이탈 사유 코드를 함께 정리한 리포트 생성
    build_reason_mapping_report(
        global_summary,
        analysis,
        XAI_OUTPUT_DIR / "reason_mapping_report.csv"
    )

    # -----------------------------------
    # 9. SHAP summary plot 저장 (beeswarm 형태)
    # -----------------------------------
    shap.summary_plot(shap_matrix, X, show=False)
    plt.tight_layout()
    plt.savefig(XAI_OUTPUT_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()

    # -----------------------------------
    # 10. SHAP bar plot 저장
    # -----------------------------------
    shap.summary_plot(shap_matrix, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(XAI_OUTPUT_DIR / "shap_bar.png", bbox_inches="tight")
    plt.close()

    # -----------------------------------
    # 11. 주요 변수 dependence plot 저장
    # -----------------------------------
    # 지정한 주요 변수에 대해 SHAP dependence plot 생성
    for feature, filename in [
        ("total_usage_count", "shap_dependence_usage.png"),
        ("error_rate", "shap_dependence_error.png")
    ]:
        # 해당 변수가 실제 데이터에 있을 때만 그래프 생성
        if feature in X.columns:
            shap.dependence_plot(feature, shap_matrix, X, show=False)
            plt.tight_layout()
            plt.savefig(XAI_OUTPUT_DIR / filename, bbox_inches="tight")
            plt.close()

    # -----------------------------------
    # 12. 저장 완료 로그 출력
    # -----------------------------------
    logger.info("saved shap outputs")


# 스크립트를 직접 실행할 때 main 함수 호출
if __name__ == "__main__":
    main()