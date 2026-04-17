# 1. ravenstack_accounts.csv (고객 계정 정보)
## 고객(기업)의 기본적인 프로필 정보
| 컬럼명 | 번역 (의미) | 설명 |
|-----------|------|-----------|
| account_id | 계정 ID | 고객을 식별하는 고유 키 (다른 파일과 연결할 때 사용) |
| account_name | 업체명 | 고객사의 이름 |
| industry | 산업군 | EdTech, FinTech, HealthTech 등 고객사가 속한 업종 |
| country | 국가 | 고객사가 위치한 국가 (US, IN, UK 등) |
| signup_date | 가입일 | 처음 서비스를 구독한 날짜 |
| referral_source | 유입 경로 | 유료 광고, 유기적 검색, 파트너십 등 가입 계기 |
| plan_tier | 요금제 등급 | Basic, Pro, Enterprise 등 현재 이용 중인 등급 |
| seats	| 시트(사용자) 수 | 해당 계정에서 사용하는 계정 수 (규모 파악 가능) |
| is_trial | 체험판 여부 | 현재 무료 체험판(Trial)을 사용 중인지 여부 |
| churn_flag | 이탈 여부 | 머신러닝의 정답(Target). True면 이탈, False면 유지 |


# 2. ravenstack_subscriptions.csv (구독 이력)
## 결제 관련 상세 정보
| 컬럼명 | 번역 (의미) | 설명 |
|-----------|------|-----------|
| subscription_id | 구독 ID | 각 구독 건당 부여되는 고유 번호 |
| account_id | 계정 ID | 어떤 고객의 구독인지 식별 |
| start_date | 구독 시작일 | 현재 구독 플랜이 시작된 날짜 |
| end_date | 구독 종료일 | 구독이 끝난 날짜 (이탈한 경우 기록됨) |
| plan_tier	| 요금제 등급 | 구독 시점의 요금제 등급 |
| seats | 시트 수 | 구독한 사용자 계정 수 |
| mrr_amount | 월간 반복 매출 | 해당 고객이 매달 내는 금액 (SaaS 핵심 지표) |
| arr_amount | 연간 반복 매출 | 해당 고객이 연간 내는 금액 (MRR x 12) |
| is_trial | 체험판 여부 | 해당 구독이 체험판인지 여부 |
| upgrade_flag | 등급 상향 여부 | 이전보다 더 비싼 요금제로 바꿨는지 여부 |
| downgrade_flag | 등급 하향 여부 | 이전보다 더 싼 요금제로 바꿨는지 여부 |
| churn_flag | 이탈 여부 | 해당 구독 건에서 이탈이 발생했는지 여부 |
| billing_frequency | 결제 주기 | monthly(월 결제) 또는 annual(연 결제) |
| auto_renew_flag | 자동 갱신 여부 | 구독이 자동으로 연장되도록 설정했는지 여부 |


# 3. ravenstack_feature_usage.csv (기능 사용 로그) / 핵심데이터
## 고객이 실제 제품을 어떻게 썼는지 보여주는 활동 데이터
| 컬럼명 | 번역 (의미) | 설명 |
|-----------|------|-----------|
| usage_id | 사용 로그 ID | 각 사용 기록의 고유 번호 |
| subscription_id | 구독 ID | 어떤 구독 상태에서 사용했는지 식별 |
| usage_date | 사용 일자 | 기능을 사용한 날짜 |
| feature_name | 기능 이름 | 어떤 기능을 사용했는지 (feature_1, feature_2 등) |
| usage_count | 사용 횟수 | 해당 날짜에 해당 기능을 몇 번 실행했는지 |
| usage_duration_secs | 사용 시간(초) | 해당 기능을 사용한 총 시간 |
| error_count | 에러 발생 횟수 | 기능 사용 중 에러가 발생한 횟수 (불만족 요인) |
| is_beta_feature | 베타 기능 여부 | 테스트 중인 기능을 사용했는지 여부 |


# 4. ravenstack_support_tickets.csv (고객 지원 티켓)
## 고객이 겪은 문제나 문의 사항
| 컬럼명 | 번역 (의미) | 설명 |
|-----------|------|-----------|
| ticket_id	| 티켓 ID | 문의 건당 고유 번호 |
| account_id | 계정 ID	어떤 고객이 문의했는지 식별 |
| submitted_at | 접수 시간	문의가 등록된 시점 |
| closed_at | 종료 시간	문의 처리가 완료된 시점 |
| resolution_time_hours | 해결 소요 시간 | 문의 접수부터 종료까지 걸린 시간(시간 단위) |
| priority | 우선순위 | urgent(긴급), high, medium, low |
| first_response_time_minutes | 첫 응답 시간	문의 후 상담원이 처음 답변하기까지 걸린 시간(분) |
| satisfaction_score | 만족도 점수 | 상담 후 고객이 매긴 점수 (중요한 예측 변수) |
| escalation_flag | 상급자 이관 여부 | 문제가 어려워 상급자에게 전달되었는지 여부 |


# 5. ravenstack_churn_events.csv (이탈 이벤트 상세)
## 이탈한 고객들에 대한 추가 정보
| 컬럼명 | 번역 (의미) | 설명 |
|-----------|------|-----------|
| churn_event_id | 이탈 이벤트 ID | 이탈 기록 고유 번호 |
| account_id | 계정 ID | 이탈한 고객 식별 |
| churn_date | 이탈 일자 | 실제로 서비스가 종료된 날짜 |
| reason_code | 이탈 사유 코드 | pricing(가격), budget(예산), features(기능 부족) 등 |
| refund_amount_usd | 환불 금액 | 이탈 시 환불해 준 금액(USD) |
| preceding_upgrade_flag | 이탈 전 상향 여부 | 이탈 직전에 요금제를 올린 적이 있는지 |
| preceding_downgrade_flag | 이탈 전 하향 여부 | 이탈 직전에 요금제를 내린 적이 있는지 |
| is_reactivation | 재활성화 여부 | 예전에 나갔다가 다시 돌아왔던 고객인지 여부 |
| feedback_text | 고객 피드백 | 고객이 직접 남긴 이탈 사유 텍스트 |
