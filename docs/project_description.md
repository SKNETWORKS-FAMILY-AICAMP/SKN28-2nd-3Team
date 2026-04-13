# 프로젝트 배경 및 문제정의

본 프로젝트는 synthetic SaaS 데이터인 RavenStack을 활용하여 account 수준 churn 여부를 예측하는 것을 목표로 한다.
`churn_events`는 이탈 이후 사유와 환불 정보가 포함될 수 있으므로 예측용 train table에는 직접 사용하지 않고 analysis table에만 연결한다.
