# save_model.py

"""
이 파일은 학습이 완료된 모델 객체를 지정한 경로에 저장하는 공통 유틸 모듈이다.

주요 역할:
- 저장 경로의 상위 폴더가 없으면 자동 생성
- joblib 형식으로 모델 객체 저장
- 이후 예측 및 재사용이 가능하도록 학습 결과를 파일로 보존
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import joblib  # 파이썬 객체 직렬화 및 저장 라이브러리


def save_model(model, path) -> None:
    # 저장할 경로의 상위 폴더가 없으면 자동 생성
    path.parent.mkdir(parents=True, exist_ok=True)

    # 모델 객체를 joblib 형식으로 파일에 저장
    joblib.dump(model, path)