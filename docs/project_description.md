# 프로젝트 배경 및 문제정의

본 프로젝트는 synthetic SaaS 데이터인 RavenStack을 활용하여 account 수준 churn 여부를 예측하는 것을 목표로 한다.
`churn_events`는 이탈 이후 사유와 환불 정보가 포함될 수 있으므로 예측용 train table에는 직접 사용하지 않고 analysis table에만 연결한다.

# 0. 프로젝트 개요

- RavenStack Synthetic SaaS Dataset 기반 고객 이탈 예측
- account 단위 데이터 통합 분석
- 예측 → 해석 → 유지 전략 연결까지 수행하는 End-to-End 프로젝트

---

## 1. 문제 정의

SaaS 환경에서는 이탈을 사후 분석하는 것보다  
**이탈 가능성이 높은 고객을 사전에 탐지하고 선제 대응하는 것**이 중요하다.

기존 접근 방식의 한계:
- 단순 사용량 기반 판단
- 사후 대응 중심 전략
- 이탈 원인에 대한 설명 부족

따라서 본 프로젝트는
**이탈 예측 + 설명 가능성 + 실행 전략 연결**을 목표로 한다.

---

## 2. 목표

- 고객 이탈 여부(`churn`) 예측 모델 개발
- Recall 중심 성능 최적화 (이탈 고객 놓치지 않기)
- SHAP 기반 이탈 원인 해석
- 실무 적용 가능한 retention 전략 도출

---

## 3. 데이터셋

- 데이터: RavenStack Synthetic SaaS Dataset
- 분석 단위: account 기준

| 항목 | 값 |
|------|----|
| 분석 고객 수 | 500명 |
| 입력 피처 수 | 74개 |
| churn 비율 | 22.0% |
| 문제 유형 | 불균형 이진 분류 |

---

## 4. 분석 흐름

1. 데이터 전처리 및 account 단위 통합
2. EDA를 통한 churn 특성 분석
3. Feature Engineering
4. ML/DL 모델 비교
5. Threshold tuning (F1 기준)
6. SHAP 기반 해석
7. 고객 유지 전략 연결

---

## 5. 모델 구조

### Machine Learning
- Logistic Regression
- Random Forest

### Deep Learning
- MLP 기반 이진 분류 모델

---

## 6. 평가 지표

- Accuracy
- Precision
- Recall (핵심 지표)
- F1-score

---

## 7. 모델 운영 기준

- 단순 accuracy가 아닌 **Recall 중심 평가**
- Threshold tuning을 통한 실무 적용 최적화
- 기본값 0.5 대신 F1 기준 threshold tuning 결과를 별도로 비교

---

## 8. 핵심 결과 요약

| 항목 | 값 |
|------|----|
| 최적 모델 | Logistic Regression |
| 최적 Threshold | 0.45 |
| F1 Score | 0.433 |

- Threshold 조정 시 성능 개선 확인
- 기본값(0.5)보다 **0.45에서 더 좋은 F1 확보**
- 이탈 고객 탐지 성능(Recall) 개선

---

## 9. 인사이트

대표 이탈 신호:
- `active_subscription_ratio`

추가 핵심 패턴:
- 활성 구독 비율 감소
- 최근 사용 감소
- 서비스 참여도 저하

단순 사용량이 아닌  
**“활성도 + 참여 패턴”이 이탈의 핵심 요인**

---

## 10. 한계

- 데이터 규모 (500 account) 제한
- 불균형 데이터 문제 존재
- synthetic 데이터 기반 → 실제 환경 검증 필요

---

## 11. 개선 방향

- 하이퍼파라미터 튜닝 고도화
- 불균형 데이터 처리 기법 적용 (SMOTE 등)
- 시계열 기반 모델 (LSTM 등) 확장
- 실제 SaaS 데이터로 검증
- 고객 세그먼트별 맞춤 전략 강화

---