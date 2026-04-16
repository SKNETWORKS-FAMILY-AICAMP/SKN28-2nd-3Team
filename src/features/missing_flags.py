# missing_flags.py

"""
이 파일은 주요 컬럼들의 결측 여부를 별도 flag 변수로 기록하고,
행동 이력이 거의 없는 고객을 식별하기 위한 보조 feature를 생성하는 모듈이다.

주요 역할:
- 설정값에 정의된 컬럼들의 결측 여부를 0/1 flag로 생성
- 사용 이력이 없는 비활성 사용자 여부 생성
- 티켓 이력이 없는 고객 여부 생성
- 결측 자체를 정보로 활용할 수 있도록 파생 변수 추가
"""
from __future__ import annotations  # 타입 힌트 지연 평가

import pandas as pd  # 데이터 처리 라이브러리
from src.config.settings import MISSING_FLAG_COLUMNS  # 결측 flag를 만들 컬럼 목록


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    # 원본 데이터 손상을 막기 위해 복사본 생성
    out = df.copy()

    # -----------------------------------
    # 1. 주요 컬럼별 결측 여부 flag 생성
    # -----------------------------------
    for col in MISSING_FLAG_COLUMNS:
        # 해당 컬럼이 실제 데이터에 존재할 때만 실행
        if col in out.columns:
            # 결측이면 1, 아니면 0으로 표시
            out[f"{col}_missing_flag"] = out[col].isna().astype(int)

    # -----------------------------------
    # 2. 비활성 사용자 여부 생성
    # -----------------------------------
    # total_usage_count가 0이면 사용 이력이 없는 사용자로 간주
    out["is_inactive_user"] = (out.get("total_usage_count", 0) == 0).astype(int)

    # -----------------------------------
    # 3. 티켓 이력 없음 여부 생성
    # -----------------------------------
    # total_tickets가 0이면 고객지원 이력이 없는 고객으로 간주
    out["has_no_ticket_history"] = (out.get("total_tickets", 0) == 0).astype(int)

    # 파생 변수가 추가된 데이터 반환
    return out