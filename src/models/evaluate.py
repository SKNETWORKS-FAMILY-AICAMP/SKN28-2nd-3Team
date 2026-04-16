# evaluate.py

"""
이 파일은 이진 분류 모델의 예측 결과를 평가하기 위한 공통 평가 모듈이다.

주요 역할:
- accuracy, precision, recall, f1, roc_auc 등 핵심 분류 성능 지표 계산
- confusion matrix를 데이터프레임 형태로 정리
- ROC curve와 Precision-Recall curve용 데이터 생성
- 평가 결과를 하나의 구조화된 객체(EvaluationResult)로 반환
"""
from __future__ import annotations  # 타입 힌트 지연 평가

from dataclasses import dataclass  # 평가 결과를 구조화된 객체로 정의하기 위한 데코레이터
import pandas as pd  # 데이터프레임 처리 라이브러리
from sklearn.metrics import (
    accuracy_score,          # 정확도 계산
    precision_score,         # 정밀도 계산
    recall_score,            # 재현율 계산
    f1_score,                # F1-score 계산
    roc_auc_score,           # ROC-AUC 계산
    confusion_matrix,        # 혼동행렬 계산
    precision_recall_curve,  # PR curve 좌표 계산
    roc_curve                # ROC curve 좌표 계산
)


@dataclass
class EvaluationResult:
    # 평가 지표들을 담는 딕셔너리
    metrics: dict

    # confusion matrix를 DataFrame 형태로 저장
    confusion_matrix_df: pd.DataFrame

    # ROC curve 시각화용 데이터
    roc_curve_df: pd.DataFrame

    # Precision-Recall curve 시각화용 데이터
    pr_curve_df: pd.DataFrame


def evaluate_binary_classifier(y_true, y_pred, y_proba) -> EvaluationResult:
    # -----------------------------------
    # 1. 핵심 분류 성능 지표 계산
    # -----------------------------------
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),                      # 전체 예측 중 맞춘 비율
        "precision": precision_score(y_true, y_pred, zero_division=0),  # 양성으로 예측한 것 중 실제 양성 비율
        "recall": recall_score(y_true, y_pred, zero_division=0),        # 실제 양성 중 맞춘 비율
        "f1": f1_score(y_true, y_pred, zero_division=0),                # precision과 recall의 조화평균
        "roc_auc": roc_auc_score(y_true, y_proba),                      # 확률 기반 분류 성능
    }

    # -----------------------------------
    # 2. confusion matrix 계산
    # -----------------------------------
    cm = confusion_matrix(y_true, y_pred)

    # 보기 쉽도록 DataFrame으로 변환
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"]
    )

    # -----------------------------------
    # 3. ROC curve 데이터 생성
    # -----------------------------------
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

    # threshold 길이와 fpr/tpr 길이가 다를 수 있으므로 None으로 길이 보정
    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": list(roc_thresholds) + [None] * (len(fpr) - len(roc_thresholds))
    })

    # -----------------------------------
    # 4. Precision-Recall curve 데이터 생성
    # -----------------------------------
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

    # 현재 코드에서는 threshold는 저장하지 않고 precision / recall만 저장
    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall
    })

    # -----------------------------------
    # 5. 평가 결과 객체로 반환
    # -----------------------------------
    return EvaluationResult(
        metrics=metrics,
        confusion_matrix_df=cm_df,
        roc_curve_df=roc_df,
        pr_curve_df=pr_df
    )