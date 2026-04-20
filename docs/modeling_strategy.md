# ML / DL / XAI 전략

- ML: Logistic Regression, RandomForest baseline
- DL: `train_table_dl.csv`를 별도로 생성해 확장 가능하도록 구성
- XAI: tree model 기준 SHAP global/local explanation 저장
  
---

## 1. 모델링 목표
1. 조기 탐지: 이탈 고위험군(Account) 선제적 식별
2. 원인 설명 (XAI): 모델의 예측 판단 근거 수치화
3. 액션 도출: 식별된 원인을 바탕으로 한 고객 유지(Retention) 전략 제안

---

## 2. 데이터 전처리 및 피처 엔지니어링
* 단위 통합: 다양한 원천 데이터(`feature_usage`, `support_tickets` 등)를 최종 비즈니스 의사결정 단위인 고객(`account_id`) 기준으로 병합 및 파생 변수 생성
* 데이터 누수(Leakage) 방지: 이탈 이후의 사후 데이터(환불, 피드백 등 `churn_events`)는 학습에서 의도적으로 배제하여 모델 신뢰성 확보
* 결측치 전략: 결측치는 중앙값(Median)으로 대체하되, '피드백 무응답' 자체가 이탈 신호일 수 있으므로 결측 여부 플래그(0/1) 를 변수로 추가

---

## 3. 데이터 분리 (Train/Valid/Test Split)
* 계층적 분할 (Stratified Split): 전체 이탈 고객 비율이 22%인 클래스 불균형(Imbalance) 상태를 고려하여, 분리된 셋 모두 동일한 22%의 비율을 유지하도록 강제 분할(8:2)

---

## 4. 모델 선정 및 파이프라인
각 모델의 장점을 살려 역할을 분담하고, 재사용 가능한 파이프라인을 구축했습니다.

* ML (Baseline):
  * Logistic Regression: 각 변수의 선형적 영향도를 직관적으로 파악하기 위한 해석 가능한 기준선 모델
  * Random Forest: 비선형 패턴 포착에 능한 트리 모델 (XAI 해석의 기준 모델로 활용)
* DL (최고 성능 및 확장성):
  * MLP (Deep Learning): 층간 복잡한 상호작용(Feature Interaction)을 포착하여 최고 예측 성능 달성 (RNN/Transformer는 데이터 특성과 규모상 제외)
  * 파이프라인 확장: 딥러닝 전용 데이터인 `train_table_dl.csv`를 별도 생성하여 향후 범주형 임베딩(Categorical Embedding) 등 딥러닝 구조 고도화가 쉽게 가능하도록 구성

---

## 5. 임계값(Threshold) 최적화
* 조정 배경: 이탈 고객 비율이 22%로 적기 때문에 기본값(0.5) 사용 시 오탐 또는 미탐 위험이 큼
* 최적화 결과: 정밀도(Precision)와 재현율(Recall)의 조화평균인 F1-Score가 최대가 되는 기준 탐색
  * 최종 선정: 모델 `DL_MLP` / 운영 최적 Threshold: 0.45 / 최고 F1-Score: 0.433

---

## 6. XAI (설명 가능한 AI) 및 비즈니스 연계
학습된 Tree Model(Random Forest)을 기준으로 SHAP global / local explanation을 저장하고, 도출된 핵심 요인을 바탕으로 액션을 제안합니다.

* 오류율(Error Rate) 상위: 기술 지원팀 우선 배정 및 긴급 복구 점검
* 활성 구독 비율(Active Sub Ratio) 하위: 사용법 가이드 재전송 및 활용 독려(온보딩)
* 응답 시간(First Response Time) 지연: CS 부서 SLA(서비스 수준) 개선 및 불만 완화 프로모션
