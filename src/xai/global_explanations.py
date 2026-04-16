# global_explanations.py

"""
이 파일은 SHAP 기반 전역 설명(global explanation) 결과를 정리하여
변수별 평균 절대 SHAP 값을 저장하는 모듈이다.

주요 역할:
- feature별 평균 절대 SHAP 값(mean absolute SHAP value) 정리
- 영향력이 큰 변수부터 내림차순 정렬
- 전역 설명 결과를 CSV 파일로 저장
- 저장된 요약 테이블을 DataFrame 형태로 반환
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

import pandas as pd  # 데이터프레임 처리 라이브러리

from src.utils.io import save_csv  # CSV 저장 함수


def save_global_shap_summary(mean_abs_shap, feature_names, output_path) -> pd.DataFrame:
    # feature 이름과 평균 절대 SHAP 값을 하나의 DataFrame으로 정리
    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)  # 영향력이 큰 순서대로 정렬

    # 정리된 전역 설명 결과를 CSV 파일로 저장
    save_csv(summary, output_path)

    # 저장된 요약 테이블을 반환
    return summary