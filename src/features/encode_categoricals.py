# encode_categoricals.py

"""
이 파일은 데이터프레임에 포함된 범주형 변수들을 one-hot encoding 방식으로 변환하여
모델이 학습 가능한 수치형 입력 형태로 바꾸는 모듈이다.

주요 역할:
- 인코딩 대상 범주형 컬럼 목록 확인
- 실제 데이터프레임에 존재하는 컬럼만 선택
- pandas의 get_dummies를 활용해 one-hot encoding 수행
- 범주형 변수를 모델 입력용 수치형 컬럼으로 확장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import pandas as pd  # 데이터 처리 라이브러리


def one_hot_encode(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    # 전달받은 범주형 컬럼 목록 중 실제 데이터프레임에 존재하는 컬럼만 선택
    existing = [c for c in categorical_cols if c in df.columns]

    # 선택된 범주형 컬럼들에 대해 one-hot encoding 수행
    # dummy_na=False: NaN을 별도 더미 컬럼으로 만들지 않음
    # drop_first=False: 첫 번째 범주를 삭제하지 않고 모두 유지
    return pd.get_dummies(df, columns=existing, dummy_na=False, drop_first=False)