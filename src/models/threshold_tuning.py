# threshold_tuning.py

"""
이 파일은 분류 모델의 예측 확률에 대해 여러 threshold를 시험해 보면서
precision, recall, f1의 변화를 비교하고 최적 threshold를 찾는 모듈이다.

주요 역할:
- 여러 threshold 후보 구간을 순차적으로 탐색
- 각 threshold별 precision, recall, f1 계산
- 가장 높은 f1 score를 만드는 threshold 선택
- threshold별 평가 결과를 CSV로 저장
- 최적 threshold 값을 반환
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import numpy as np  # 수치 계산 라이브러리
import pandas as pd  # 데이터 처리 라이브러리
from sklearn.metrics import precision_score, recall_score, f1_score  # 분류 성능 지표

from src.utils.io import save_csv  # CSV 저장 함수


def tune_threshold(y_true, y_proba, output_path) -> float:
    # threshold별 평가 결과를 저장할 리스트
    rows = []

    # 최적 threshold 초기값
    best_threshold = 0.5

    # 현재까지 가장 높은 f1 score 초기화
    best_f1 = -1

    # -----------------------------------
    # 1. threshold 후보를 순차적으로 탐색
    # -----------------------------------
    for threshold in np.arange(0.1, 0.95, 0.05):
        # 현재 threshold 기준으로 확률값을 0/1 예측값으로 변환
        pred = (y_proba >= threshold).astype(int)

        # precision 계산
        precision = precision_score(y_true, pred, zero_division=0)

        # recall 계산
        recall = recall_score(y_true, pred, zero_division=0)

        # f1 score 계산
        f1 = f1_score(y_true, pred, zero_division=0)

        # threshold별 성능 결과 저장
        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        # 현재 threshold의 f1이 가장 좋으면 최적값 갱신
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # -----------------------------------
    # 2. threshold별 평가 결과 저장
    # -----------------------------------
    save_csv(pd.DataFrame(rows), output_path)

    # -----------------------------------
    # 3. 최적 threshold 반환
    # -----------------------------------
    return float(best_threshold)