# train_baseline.py

"""
이 파일은 로지스틱 회귀(Logistic Regression)를 baseline 모델로 학습하고,
검증 데이터 기준으로 최적 threshold를 찾은 뒤 성능 평가와 산출물 저장까지 수행하는 모듈이다.

주요 역할:
- train / valid 데이터 로드
- 입력 변수(X)와 타겟 변수(y) 분리
- 결측치 대체 및 표준화를 포함한 전처리 파이프라인 구성
- Logistic Regression baseline 모델 학습
- validation 데이터에서 최적 threshold 탐색
- 분류 성능 평가, confusion matrix / ROC / PR curve 저장
- 학습된 모델과 평가 결과를 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리
from sklearn.compose import ColumnTransformer  # 컬럼별 전처리 파이프라인 구성
from sklearn.pipeline import Pipeline  # 전처리 + 모델을 하나로 묶는 파이프라인
from sklearn.preprocessing import StandardScaler  # 수치형 변수 표준화
from sklearn.impute import SimpleImputer  # 결측값 대체
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 분류 모델

from src.config.paths import PROCESSED_DIR, MODELS_OUTPUT_DIR  # 데이터 및 산출물 경로
from src.config.settings import TARGET_COL, RANDOM_STATE  # 타겟 변수명, 난수 고정값
from src.models.evaluate import evaluate_binary_classifier  # 분류 성능 평가 함수
from src.models.threshold_tuning import tune_threshold  # 최적 threshold 탐색 함수
from src.models.save_model import save_model  # 모델 저장 함수
from src.utils.io import read_csv, save_csv  # CSV 입출력 함수
from src.utils.plot_utils import apply_plot_style  # 공통 시각화 스타일 적용 함수
from src.utils.logger import get_logger  # 로거 생성 함수

# 현재 파일 기준 logger 생성
logger = get_logger(__name__)


def _plot_confusion(cm_df: pd.DataFrame, output_path):
    # confusion matrix 시각화를 위한 figure 생성
    plt.figure(figsize=(5, 4))

    # confusion matrix 값을 이미지 형태로 표현
    plt.imshow(cm_df.values, aspect="auto")

    # 각 셀 중앙에 실제 숫자 값 표시
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, int(cm_df.iloc[i, j]), ha="center", va="center")

    # 축 라벨 설정
    plt.xticks(range(cm_df.shape[1]), cm_df.columns)
    plt.yticks(range(cm_df.shape[0]), cm_df.index)

    # 그래프 제목 설정
    plt.title("Confusion Matrix")

    # 레이아웃 자동 정리
    plt.tight_layout()

    # 이미지 파일 저장
    plt.savefig(output_path, bbox_inches="tight")

    # 메모리 정리를 위해 figure 닫기
    plt.close()


def main() -> None:
    # -----------------------------------
    # 1. 학습 / 검증 데이터 로드
    # -----------------------------------
    train = read_csv(PROCESSED_DIR / "train.csv")
    valid = read_csv(PROCESSED_DIR / "valid.csv")

    # 입력 변수(X)와 타겟 변수(y) 분리
    # account_id는 식별자이므로 모델 입력에서 제외
    X_train = train.drop(columns=[TARGET_COL, "account_id"], errors="ignore")
    y_train = train[TARGET_COL]

    X_valid = valid.drop(columns=[TARGET_COL, "account_id"], errors="ignore")
    y_valid = valid[TARGET_COL]

    # -----------------------------------
    # 2. 전처리기(preprocessor) 구성
    # -----------------------------------
    # 모든 입력 컬럼을 수치형으로 보고,
    # 결측값은 중앙값으로 대체한 뒤 표준화 수행
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  # 결측값 중앙값 대체
            ("scaler", StandardScaler()),                   # 평균 0, 표준편차 1로 표준화
        ]), X_train.columns.tolist())
    ])

    # -----------------------------------
    # 3. baseline 모델 파이프라인 구성
    # -----------------------------------
    model = Pipeline([
        ("preprocessor", preprocessor),  # 전처리 단계
        ("classifier", LogisticRegression(
            max_iter=2000,               # 반복 횟수 충분히 확보
            class_weight="balanced",     # 클래스 불균형 보정
            random_state=RANDOM_STATE    # 재현성 확보
        )),
    ])

    # -----------------------------------
    # 4. 모델 학습
    # -----------------------------------
    model.fit(X_train, y_train)

    # -----------------------------------
    # 5. validation 예측 및 threshold tuning
    # -----------------------------------
    # validation 데이터에 대한 양성 클래스 확률 예측
    valid_proba = model.predict_proba(X_valid)[:, 1]

    # F1 기준 최적 threshold 탐색
    best_threshold = tune_threshold(
        y_valid,
        valid_proba,
        MODELS_OUTPUT_DIR / "threshold_metrics.csv"
    )

    # 최적 threshold 기준으로 최종 예측 라벨 생성
    valid_pred = (valid_proba >= best_threshold).astype(int)

    # -----------------------------------
    # 6. 성능 평가
    # -----------------------------------
    result = evaluate_binary_classifier(y_valid, valid_pred, valid_proba)

    # -----------------------------------
    # 7. 평가 결과 테이블 저장
    # -----------------------------------
    # baseline 성능 요약 저장
    metrics_df = pd.DataFrame([{
        "model": "logistic_regression",
        "best_threshold": best_threshold,
        **result.metrics
    }])
    save_csv(metrics_df, MODELS_OUTPUT_DIR / "baseline_metrics.csv")

    # confusion matrix 저장
    save_csv(result.confusion_matrix_df.reset_index(), MODELS_OUTPUT_DIR / "confusion_matrix.csv")

    # ROC curve 좌표 저장
    save_csv(result.roc_curve_df, MODELS_OUTPUT_DIR / "roc_curve_points.csv")

    # PR curve 좌표 저장
    save_csv(result.pr_curve_df, MODELS_OUTPUT_DIR / "pr_curve_points.csv")

    # 학습된 모델 저장
    save_model(model, MODELS_OUTPUT_DIR / "baseline_model.pkl")

    # -----------------------------------
    # 8. 시각화 저장
    # -----------------------------------
    apply_plot_style()

    # confusion matrix 이미지 저장
    _plot_confusion(result.confusion_matrix_df, MODELS_OUTPUT_DIR / "confusion_matrix.png")

    # ROC curve 시각화 및 저장
    plt.figure()
    plt.plot(result.roc_curve_df["fpr"], result.roc_curve_df["tpr"])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(MODELS_OUTPUT_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()

    # PR curve 시각화 및 저장
    plt.figure()
    plt.plot(result.pr_curve_df["recall"], result.pr_curve_df["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.tight_layout()
    plt.savefig(MODELS_OUTPUT_DIR / "pr_curve.png", bbox_inches="tight")
    plt.close()

    # 저장 완료 로그 출력
    logger.info("saved baseline model outputs")


# 스크립트를 직접 실행할 때 main 함수 호출
if __name__ == "__main__":
    main()