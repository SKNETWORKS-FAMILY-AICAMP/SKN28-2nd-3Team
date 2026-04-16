# train_tree_model.py

"""
이 파일은 Random Forest 기반 트리 모델을 학습하고,
validation 데이터에서 최적 threshold를 찾은 뒤 성능 평가와 feature importance 저장까지 수행하는 모듈이다.

주요 역할:
- train / valid 데이터 로드
- 입력 변수(X)와 타겟 변수(y) 분리
- Random Forest 분류 모델 학습
- validation 데이터 기준 최적 threshold 탐색
- 분류 성능 평가 및 비교표 저장
- feature importance 계산 및 시각화
- 학습된 모델과 평가 결과를 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 분류 모델

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
    # 2. Random Forest 모델 정의 및 학습
    # -----------------------------------
    model = RandomForestClassifier(
        n_estimators=300,                  # 생성할 트리 개수
        min_samples_leaf=2,                # 리프 노드 최소 샘플 수
        class_weight="balanced_subsample", # 클래스 불균형 보정
        random_state=RANDOM_STATE,         # 재현성 확보
        n_jobs=-1,                         # 가능한 모든 CPU 코어 사용
    )

    # 모델 학습
    model.fit(X_train, y_train)

    # -----------------------------------
    # 3. validation 예측 및 threshold tuning
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
    # 4. 성능 평가
    # -----------------------------------
    result = evaluate_binary_classifier(y_valid, valid_pred, valid_proba)

    # 현재 랜덤 포레스트 모델 성능 요약
    current = pd.DataFrame([{
        "model": "random_forest",
        "best_threshold": best_threshold,
        **result.metrics
    }])

    # -----------------------------------
    # 5. 기존 baseline과 비교표 생성
    # -----------------------------------
    baseline_path = MODELS_OUTPUT_DIR / "baseline_metrics.csv"

    # baseline 결과가 있으면 이어 붙이고, 없으면 현재 결과만 사용
    comparison = (
        pd.concat([read_csv(baseline_path), current], ignore_index=True)
        if baseline_path.exists()
        else current
    )

    # 모델 비교표 저장
    save_csv(comparison, MODELS_OUTPUT_DIR / "model_comparison.csv")

    # confusion matrix 저장
    save_csv(result.confusion_matrix_df.reset_index(), MODELS_OUTPUT_DIR / "confusion_matrix_tree.csv")

    # ROC curve 좌표 저장
    save_csv(result.roc_curve_df, MODELS_OUTPUT_DIR / "roc_curve_points_tree.csv")

    # PR curve 좌표 저장
    save_csv(result.pr_curve_df, MODELS_OUTPUT_DIR / "pr_curve_points_tree.csv")

    # -----------------------------------
    # 6. feature importance 계산 및 저장
    # -----------------------------------
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # 변수 중요도 테이블 저장
    save_csv(importances, MODELS_OUTPUT_DIR / "feature_importance.csv")

    # 학습된 모델 저장
    save_model(model, MODELS_OUTPUT_DIR / "best_model.pkl")

    # -----------------------------------
    # 7. feature importance 시각화
    # -----------------------------------
    apply_plot_style()

    plt.figure(figsize=(10, 6))

    # 상위 20개 변수 중요도를 가로 막대그래프로 시각화
    plt.barh(
        importances.head(20).sort_values("importance")["feature"],
        importances.head(20).sort_values("importance")["importance"]
    )

    # 그래프 제목 설정
    plt.title("Feature Importance")

    # 레이아웃 자동 정리
    plt.tight_layout()

    # 이미지 저장
    plt.savefig(MODELS_OUTPUT_DIR / "feature_importance.png", bbox_inches="tight")

    # 메모리 정리를 위해 figure 닫기
    plt.close()

    # 저장 완료 로그 출력
    logger.info("saved tree model outputs")


# 스크립트를 직접 실행할 때 main 함수 호출
if __name__ == "__main__":
    main()