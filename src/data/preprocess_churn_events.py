# preprocess_churn_events.py

"""
이 파일은 churn_events(고객 이탈 이벤트) 원천 데이터를 전처리하여
이탈 발생 시점, 직전 상태 변화, 이탈 사유 등의 정보를 분석 가능한 형태로 정제하는 모듈이다.

주요 역할:
- 원천 churn 이벤트 데이터 로드
- 이탈 날짜 컬럼 datetime 변환
- boolean 컬럼을 0/1로 변환
- 이탈 사유 코드 및 고객 피드백 결측값 처리
- 전처리된 결과를 interim 영역에 저장
"""
from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정

import pandas as pd  # 데이터프레임 처리 라이브러리

# 경로 설정 (raw → interim)
from src.config.paths import RAW_DIR, INTERIM_DIR

# 설정값 (파일명)
from src.config.settings import CHURN_EVENTS_FILE

# 유틸 함수 (날짜 변환, bool → int 변환)
from src.utils.helpers import to_datetime, bool_to_int

# 입출력 및 로깅
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

# logger 생성
logger = get_logger(__name__)


# -----------------------------------
# churn 이벤트 데이터 전처리
# -----------------------------------
def preprocess_churn_events() -> pd.DataFrame:

    # 원천 churn 이벤트 데이터 로드
    df = read_csv(RAW_DIR / CHURN_EVENTS_FILE)

    # churn_date를 datetime으로 변환
    df = to_datetime(df, ["churn_date"])

    # boolean 컬럼들을 0/1로 변환
    for col in ["preceding_upgrade_flag", "preceding_downgrade_flag", "is_reactivation"]:
        df[col] = bool_to_int(df[col])

    # 이탈 사유 코드 결측값 처리
    df["reason_code"] = df["reason_code"].fillna("unknown")

    # 고객 피드백 텍스트 결측값 처리
    df["feedback_text"] = df["feedback_text"].fillna("no_feedback")

    # 정제된 데이터 반환
    return df


# -----------------------------------
# 메인 실행 함수
# -----------------------------------
def main() -> None:
    df = preprocess_churn_events()  # 전처리 수행

    # interim 경로에 저장
    save_csv(df, INTERIM_DIR / "churn_events_clean.csv")

    # 로그 출력 (데이터 shape 포함)
    logger.info("saved churn_events_clean.csv shape=%s", df.shape)


# 스크립트 실행 시 main 함수 호출
if __name__ == "__main__":
    main()