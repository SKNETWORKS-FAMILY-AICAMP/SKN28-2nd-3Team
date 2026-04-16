# reason_mapping.py

"""
이 파일은 모델이 중요하게 본 전역 설명 결과와
실제 churn 이벤트 사유 코드를 함께 정리하여 해석용 리포트를 만드는 모듈이다.

주요 역할:
- analysis_table에서 최신 이탈 사유 코드별 빈도 집계
- global SHAP summary에서 상위 중요 feature 추출
- 모델 관점의 주요 신호와 실제 이탈 사유를 하나의 테이블로 결합
- 해석용 reason mapping 리포트를 CSV 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

import pandas as pd  # 데이터프레임 처리 라이브러리

from src.utils.io import save_csv  # CSV 저장 함수


def build_reason_mapping_report(global_summary: pd.DataFrame, analysis_table: pd.DataFrame, output_path) -> None:
    # analysis_table에서 최신 이탈 사유 코드를 기준으로 빈도 집계
    # 결측값은 "unknown"으로 대체하여 누락 없이 집계
    reason_counts = (
        analysis_table["latest_reason_code"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("reason_code")
        .reset_index(name="count")
    )

    # 전역 SHAP 요약에서 상위 10개 중요 변수만 추출
    top_features = global_summary.head(10).copy()

    # 상위 중요 변수에 설명용 note 추가
    top_features["note"] = "모델이 중요하게 본 이탈 전 신호"

    # 실제 이탈 사유 코드 집계 결과에도 설명용 note 추가
    reason_counts["note"] = "실제 churn_events 기반 사후 사유 요약"

    # 모델 중요 변수와 실제 이탈 사유를 같은 형식으로 맞춰 하나의 테이블로 결합
    combined = pd.concat([
        top_features.rename(columns={"feature": "item", "mean_abs_shap": "score"})[["item", "score", "note"]],
        reason_counts.rename(columns={"reason_code": "item", "count": "score"})[["item", "score", "note"]],
    ], ignore_index=True)

    # 최종 reason mapping 리포트를 CSV 파일로 저장
    save_csv(combined, output_path)