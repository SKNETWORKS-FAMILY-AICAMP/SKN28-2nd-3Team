# local_explanations.py

"""
이 파일은 개별 샘플에 대한 SHAP 기반 로컬 설명(local explanation) 결과를 저장하는 모듈이다.

주요 역할:
- 입력 데이터에서 설명할 샘플 인덱스 선택
- 각 샘플별 feature 값과 shap 값을 하나의 테이블로 정리
- 절대 shap 값이 큰 순서대로 정렬하여 중요한 변수부터 확인 가능하게 구성
- 샘플별 로컬 설명 결과를 CSV 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

import pandas as pd  # 데이터프레임 처리 라이브러리

from src.utils.io import save_csv  # CSV 저장 함수


def save_local_explanations(shap_values, X: pd.DataFrame, output_prefix) -> None:
    # 로컬 설명을 저장할 샘플 인덱스 선택
    # 기본적으로 첫 번째 샘플(0)과, 데이터가 2개 이상이면 두 번째 샘플(1)을 사용
    # 데이터가 1개뿐이면 [0, 0]이 되어 같은 샘플이 두 번 저장될 수 있음
    sample_idx = [0, min(1, len(X) - 1)]

    # 선택한 샘플 인덱스를 하나씩 순회하며 로컬 설명 결과 생성
    for i, idx in enumerate(sample_idx, start=1):
        # 현재 샘플의 feature 값과 shap 값을 하나의 DataFrame으로 정리
        local = pd.DataFrame({
            "feature": X.columns,                # 변수명
            "feature_value": X.iloc[idx].values, # 해당 샘플의 실제 변수값
            "shap_value": shap_values[idx],      # 해당 샘플의 shap 값
        }).sort_values(
            "shap_value",
            key=lambda s: s.abs(),               # 절대 shap 값 기준으로 정렬
            ascending=False                      # 영향력이 큰 변수부터 위로 오도록 내림차순 정렬
        )

        # 샘플별 로컬 설명 결과를 CSV 파일로 저장
        save_csv(local, output_prefix.parent / f"local_explanation_sample_{i}.csv")