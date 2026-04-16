# helpers.py

"""
이 파일은 전처리 과정에서 반복적으로 사용하는 공통 helper 함수를 모아둔 유틸 모듈이다.

주요 역할:
- 지정한 컬럼들을 datetime 형식으로 안전하게 변환
- boolean 값을 0/1 정수형으로 변환
- 0으로 나누는 상황을 방지하며 안전하게 나눗셈 수행
- 결측값 또는 특정 sentinel 값을 기준으로 missing flag 생성
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

from typing import Iterable  # 여러 컬럼명을 받을 때 사용할 타입 힌트
import numpy as np  # 수치 계산 라이브러리
import pandas as pd  # 데이터프레임 처리 라이브러리


def to_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    # 전달받은 컬럼 목록을 하나씩 순회
    for col in columns:
        # 해당 컬럼이 실제 데이터프레임에 있을 때만 변환 수행
        if col in df.columns:
            # 날짜 변환이 불가능한 값은 NaT로 처리하여 안전하게 변환
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # datetime 변환이 반영된 데이터프레임 반환
    return df


def bool_to_int(series: pd.Series) -> pd.Series:
    # 결측값은 False로 간주한 뒤, boolean 값을 0/1 정수형으로 변환
    return series.fillna(False).astype(int)


def safe_divide(numerator, denominator, fill_value: float = 0.0):
    # 분자 데이터를 numpy 배열로 변환
    numerator = np.asarray(numerator)

    # 분모 데이터도 numpy 배열로 변환
    denominator = np.asarray(denominator)

    # 분모가 0이면 fill_value를 사용하고, 아니면 일반 나눗셈 수행
    return np.where(denominator == 0, fill_value, numerator / denominator)


def create_missing_flag(series: pd.Series, sentinel=None) -> pd.Series:
    # sentinel 값이 지정되지 않으면 일반적인 결측 여부만 0/1로 반환
    if sentinel is None:
        return series.isna().astype(int)

    # sentinel 값이 지정되면, 결측이거나 sentinel과 같은 경우를 모두 missing으로 간주
    return (series.isna() | (series == sentinel)).astype(int)