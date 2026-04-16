# make_dl_dataset.py

"""
이 파일은 머신러닝용 학습 테이블을 딥러닝 모델 입력에 적합한 형태로 변환하는 모듈이다.

주요 역할:
- 원본 데이터프레임 복사
- 타겟 변수(churn_flag) 분리
- 수치형 컬럼 선택
- StandardScaler를 적용해 수치형 변수 표준화
- 표준화된 데이터에 타겟 변수를 다시 결합
- 딥러닝 입력용 테이블을 CSV로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import pandas as pd  # 데이터 처리 라이브러리
from sklearn.preprocessing import StandardScaler  # 수치형 변수 표준화 도구

from src.utils.io import save_csv  # CSV 저장 함수


def make_dl_ready_table(df: pd.DataFrame, output_path) -> pd.DataFrame:
    # 원본 데이터 손상을 막기 위해 복사본 생성
    out = df.copy()

    # 타겟 변수 보관용 변수 초기화
    target = None

    # -----------------------------------
    # 1. 타겟 변수 분리
    # -----------------------------------
    # churn_flag가 존재하면 따로 분리해둠
    if "churn_flag" in out.columns:
        target = out["churn_flag"].copy()
        out = out.drop(columns=["churn_flag"])

    # -----------------------------------
    # 2. 수치형 컬럼 선택
    # -----------------------------------
    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()

    # -----------------------------------
    # 3. 수치형 변수 표준화
    # -----------------------------------
    # 평균 0, 표준편차 1 기준으로 스케일 조정
    scaler = StandardScaler()
    out[numeric_cols] = scaler.fit_transform(out[numeric_cols])

    # -----------------------------------
    # 4. 타겟 변수 다시 결합
    # -----------------------------------
    if target is not None:
        out["churn_flag"] = target.values

    # -----------------------------------
    # 5. 결과 저장
    # -----------------------------------
    save_csv(out, output_path)

    # 표준화가 완료된 딥러닝용 데이터 반환
    return out