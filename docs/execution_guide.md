# Execution Guide

아래 순서대로 실행하면  
원천 데이터 확인 → 전처리 → 통합 테이블 생성 → 피처 엔지니어링 → 데이터 분리 → 모델 학습 → XAI 분석 → Streamlit 실행까지 한 번에 이어서 진행할 수 있다.

> **Note**
> README에는 `requirements.txt` 설치 및 Streamlit 실행 명령어만 간단히 작성하고, 상세 실행 절차는 본 문서에서 별도로 안내한다.

---

## 0. 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 1. 원천 데이터 점검

`data/raw/` 폴더의 5개 CSV 파일을 기준으로 컬럼 타입, 결측치, 고유값 등을 점검한다.  
결과는 `docs/raw_data_check_summary.csv`에 저장된다.

```bash
python -m src.data.data_check
```

---

## 2. 테이블별 전처리

각 원천 테이블을 정제하여 `data/interim/` 폴더에 저장한다.

```bash
python -m src.data.preprocess_accounts
python -m src.data.preprocess_subscriptions
python -m src.data.preprocess_feature_usage
python -m src.data.preprocess_support_tickets
python -m src.data.preprocess_churn_events
```

### 생성 예시

- `data/interim/accounts_clean.csv`
- `data/interim/subscriptions_clean.csv`
- `data/interim/feature_usage_agg.csv`
- `data/interim/support_tickets_agg.csv`
- `data/interim/churn_events_clean.csv`

---

## 3. 통합 분석 테이블 생성

전처리된 데이터를 `account_id` 기준으로 통합하여 기본 분석 테이블을 생성한다.

```bash
python -m src.data.make_train_table
python -m src.data.make_analysis_table
```

### 생성 예시

- `data/interim/merged_base_table.csv`
- `data/processed/analysis_table.csv`

---

## 4. 피처 엔지니어링 및 범주형 인코딩

모델 학습에 사용할 파생변수를 생성하고, 범주형 변수를 인코딩한다.

```bash
python -m src.features.subscription_change_features
python -m src.features.build_features
python -m src.features.encode_categoricals
```

---

## 5. 학습용 데이터셋 분리

학습(`train`) / 검증(`valid`) / 테스트(`test`) 데이터셋으로 분리한다.

```bash
python -m src.data.split_dataset
```

### 생성 예시

- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/processed/test.csv`

---

## 6. 머신러닝 모델 학습

먼저 베이스라인(Logistic Regression) 모델을 학습한 뒤,  
트리 기반 모델(Random Forest)을 학습하여 성능을 비교한다.

```bash
python -m src.models.train_baseline
python -m src.models.train_tree_model
```

### 주요 결과 저장 위치

- `outputs/models/baseline_metrics.csv`
- `outputs/models/model_comparison.csv`
- `outputs/models/best_model.pkl`
- `outputs/models/feature_importance.csv`

---

## 7. 딥러닝 모델 학습

MLP 기반 딥러닝 모델을 학습하고,  
추후 예측에 사용할 scaler 및 feature 목록도 함께 저장한다.

```bash
python -m src.features.make_dl_dataset
python -m src.models.train_dl_model
python -m src.models.predict_dl_model
```

### 주요 결과 저장 위치

- `outputs/models/dl_model.pth`
- `outputs/models/dl_feature_columns.csv`
- `outputs/models/dl_scaler.pkl`
- `outputs/models/dl_test_predictions.csv`

---

## 8. XAI 분석

학습된 최종 트리 모델을 바탕으로 SHAP 분석을 수행하여 주요 이탈 요인을 해석한다.

```bash
python -m src.xai.shap_analysis
```

### 주요 결과 저장 위치

- `outputs/xai/shap_summary.png`
- `outputs/xai/shap_bar.png`
- `outputs/xai/xai_summary_report.csv`
- `outputs/xai/reason_mapping_report.csv`

---

## 9. Streamlit 대시보드 실행

최종적으로 Streamlit 대시보드를 실행하여  
EDA, 모델 성능, XAI 결과, 고객별 예측을 확인한다.

```bash
streamlit run src/app/streamlit_app.py
```

---

## 전체 실행 순서 요약

```bash
python -m src.data.data_check

python -m src.data.preprocess_accounts
python -m src.data.preprocess_subscriptions
python -m src.data.preprocess_feature_usage
python -m src.data.preprocess_support_tickets
python -m src.data.preprocess_churn_events

python -m src.data.make_train_table
python -m src.data.make_analysis_table

python -m src.features.subscription_change_features
python -m src.features.build_features
python -m src.features.encode_categoricals

python -m src.data.split_dataset

python -m src.models.train_baseline
python -m src.models.train_tree_model

python -m src.features.make_dl_dataset
python -m src.models.train_dl_model
python -m src.models.predict_dl_model

python -m src.xai.shap_analysis

streamlit run src/app/streamlit_app.py
```
