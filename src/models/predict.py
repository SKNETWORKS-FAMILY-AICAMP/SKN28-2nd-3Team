# predict.py

"""
이 파일은 저장된 머신러닝 모델을 불러와
새로운 입력 데이터에 대한 예측 확률을 계산하는 공통 예측 모듈이다.

주요 역할:
- joblib 형식으로 저장된 모델 로드
- 모델이 predict_proba를 지원하면 양성 클래스 확률 반환
- predict_proba가 없는 경우 decision_function 결과 반환
- 서로 다른 모델 유형에 공통으로 사용할 수 있는 예측 인터페이스 제공
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import pandas as pd  # 데이터프레임 처리 라이브러리
import joblib  # 저장된 모델 파일 로드용 라이브러리


def load_model(path):
    # 지정한 경로의 joblib 모델 파일을 불러와 반환
    return joblib.load(path)


def predict_proba(model, X: pd.DataFrame):
    # 모델이 predict_proba 메서드를 지원하면
    # 양성 클래스(보통 1번 클래스) 확률만 반환
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    # predict_proba가 없는 모델이면 decision_function 결과 반환
    # 예: 일부 선형 모델, SVM 계열 등
    decision = model.decision_function(X)
    return decision