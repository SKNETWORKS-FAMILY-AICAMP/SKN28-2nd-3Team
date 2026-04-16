# settings.py

# -------------------------------
# 데이터 분할 및 재현성 설정
# -------------------------------

RANDOM_STATE = 42        # 모델 학습 시 랜덤성을 고정하여 결과 재현 가능하게 설정
VALID_SIZE = 0.15        # 전체 데이터 중 validation 데이터 비율
TEST_SIZE = 0.15         # 전체 데이터 중 test 데이터 비율


# -------------------------------
# 원천 데이터 파일명 정의
# -------------------------------

ACCOUNT_FILE = "accounts.csv"               # 계정(고객) 정보 파일
SUBSCRIPTIONS_FILE = "subscriptions.csv"   # 구독 정보 파일
FEATURE_USAGE_FILE = "feature_usage.csv"   # 기능 사용 로그 데이터
SUPPORT_TICKETS_FILE = "support_tickets.csv"  # 고객 문의/지원 티켓 데이터
CHURN_EVENTS_FILE = "churn_events.csv"     # 이탈 이벤트 데이터


# -------------------------------
# 주요 컬럼명 정의 (키 및 타겟)
# -------------------------------

TARGET_COL = "churn_flag"        # 예측 대상 변수 (이탈 여부: 0/1)
ACCOUNT_KEY = "account_id"       # 고객 단위 식별 키
SUBSCRIPTION_KEY = "subscription_id"  # 구독 단위 식별 키


# -------------------------------
# 범주형 변수 처리 기준
# -------------------------------

LOW_CARDINALITY_THRESHOLD = 25  # 범주 개수가 이 값 이하일 경우 low-cardinality로 간주
TOP_K_COUNTRIES_FOR_DISPLAY = 10  # 시각화 시 상위 국가 몇 개까지 표시할지


# -------------------------------
# 결측치 처리 기준
# -------------------------------

MEDIAN_IMPUTE_COLUMNS = [
    "avg_resolution_time_hours",        # 평균 문제 해결 시간
    "avg_first_response_time_minutes",  # 평균 첫 응답 시간
    "avg_satisfaction_score",           # 평균 고객 만족도 점수
]
# 위 컬럼들은 결측치 발생 시 중앙값(median)으로 대체


MISSING_FLAG_COLUMNS = [
    "avg_resolution_time_hours",
    "avg_first_response_time_minutes",
    "avg_satisfaction_score",
]
# 결측치 존재 여부 자체를 하나의 feature(플래그 변수)로 추가할 컬럼들


# -------------------------------
# 주요 수치형 feature 리스트
# -------------------------------

KEY_NUMERIC_FEATURES = [
    "account_age_days",                  # 계정 생성 후 경과 일수
    "total_subscriptions",               # 전체 구독 수
    "active_subscriptions",              # 현재 활성 구독 수
    "avg_sub_seats",                     # 평균 좌석 수
    "avg_mrr_amount",                    # 평균 월 반복 매출 (MRR)
    "total_arr_amount",                  # 총 연 반복 매출 (ARR)
    "total_usage_count",                 # 전체 사용 횟수
    "total_usage_duration_secs",         # 전체 사용 시간
    "total_error_count",                 # 총 오류 발생 횟수
    "unique_feature_count",              # 사용한 기능 종류 수
    "days_since_last_usage",             # 마지막 사용 이후 경과 일수
    "error_rate",                        # 오류 발생 비율
    "total_tickets",                     # 총 고객 문의 수
    "avg_resolution_time_hours",         # 평균 문제 해결 시간
    "avg_first_response_time_minutes",   # 평균 첫 응답 시간
    "avg_satisfaction_score",            # 평균 고객 만족도
    "escalation_ratio",                  # 이슈 escalated 비율
    "usage_per_subscription",            # 구독당 사용량
    "ticket_per_subscription",           # 구독당 문의 수
    "error_per_subscription",            # 구독당 오류 수
    "health_score",                      # 종합 고객 상태 점수
]