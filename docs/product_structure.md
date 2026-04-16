# Project Structure

프로젝트 전체 폴더 구조와 파일 역할은 아래와 같다.

> **Note**
> `.gitignore`에 포함된 `__pycache__/`, `*.pyc` 등 실행 중 자동 생성되는 캐시 파일은 아래 구조도에서 제외하였다.

---

## Directory Tree

```text
아카이브/
├── README.md                                      # 프로젝트 소개, 개요, 실행 흐름, 배포 링크 정리 문서
├── requirements.txt                               # 프로젝트 실행에 필요한 파이썬 패키지 목록

├── assets/                                        # README·발표·앱에서 사용하는 정적 리소스 폴더
│   └── images/                                    # 프로젝트 이미지 파일 모음
│       ├── 3team-image-horizontal.png             # 팀 소개용 가로형 대표 이미지
│       ├── 3team-member.png                       # 팀원 소개 이미지
│       ├── dashboard_main.png                     # 대시보드 메인 화면 이미지
│       ├── pipeline-diagram.png                   # 전체 분석/서비스 파이프라인 도식 이미지
│       ├── recommendation.png                     # 추천/전략 제안 관련 시각 자료
│       └── threshold_change.png                   # threshold 변화에 따른 성능 비교 이미지

├── data/                                          # 데이터 저장 폴더
│   ├── raw/                                       # 원천 데이터(가공 전)
│   │   ├── accounts.csv                           # 고객 계정 기본 정보 및 churn 타깃 원본 데이터
│   │   ├── churn_events.csv                       # 이탈 이벤트 및 원인 원본 데이터
│   │   ├── feature_usage.csv                      # 기능 사용 로그 원본 데이터
│   │   ├── subscriptions.csv                      # 구독 이력 원본 데이터
│   │   └── support_tickets.csv                    # 고객 지원/문의 이력 원본 데이터
│   ├── interim/                                   # 전처리 중간 산출물 저장 폴더
│   │   ├── accounts_clean.csv                     # accounts 전처리 결과
│   │   ├── churn_events_clean.csv                 # churn_events 전처리 결과
│   │   ├── feature_usage_agg.csv                  # feature_usage를 account 수준으로 집계한 결과
│   │   ├── merged_base_table.csv                  # 여러 테이블을 account 기준으로 병합한 기본 테이블
│   │   ├── subscriptions_clean.csv                # subscriptions 전처리 결과
│   │   └── support_tickets_agg.csv                # support_tickets 집계 결과
│   └── processed/                                 # 모델링용 최종 가공 데이터
│       ├── X_test.csv                             # 테스트용 입력 변수 데이터
│       ├── X_train.csv                            # 학습용 입력 변수 데이터
│       ├── X_valid.csv                            # 검증용 입력 변수 데이터
│       ├── analysis_table.csv                     # 분석 및 시각화용 최종 테이블
│       ├── test.csv                               # 테스트용 전체 데이터셋
│       ├── train.csv                              # 학습용 전체 데이터셋
│       ├── train_table_dl.csv                     # 딥러닝 학습용 테이블
│       ├── train_table_ml.csv                     # 머신러닝 학습용 테이블
│       ├── valid.csv                              # 검증용 전체 데이터셋
│       ├── y_test.csv                             # 테스트용 타깃 값
│       ├── y_train.csv                            # 학습용 타깃 값
│       └── y_valid.csv                            # 검증용 타깃 값

├── docs/                                          # 프로젝트 설명 문서 및 보조 자료
│   ├── data_dictionary.md                         # 변수/컬럼 설명 문서
│   ├── execution_guide.txt                        # 전체 코드 실행 순서 및 명령어 안내
│   ├── modeling_strategy.md                       # 모델링 전략 및 접근 방식 설명 문서
│   ├── project_description.md                     # 프로젝트 설명서
│   ├── raw_data_check_summary.csv                 # 원천 데이터 점검 결과 요약표
│   └── streamlit_guide.md                         # Streamlit 실행/구성 관련 가이드

├── notebooks/                                     # 분석 노트북 모음
│   ├── 01_data_check.ipynb                        # 원천 데이터 구조 및 품질 점검 노트북
│   ├── 02_feature_engineering.ipynb               # 피처 엔지니어링 실험 노트북
│   ├── 03_eda.ipynb                               # 탐색적 데이터 분석 노트북
│   ├── 04_ml_baseline.ipynb                       # 머신러닝 베이스라인 모델링 노트북
│   ├── 05_xai_analysis.ipynb                      # XAI/SHAP 해석 노트북
│   └── 06_dl_experiment.ipynb                     # 딥러닝 실험 노트북

├── outputs/                                       # 분석/모델링 결과물 저장 폴더
│   ├── eda/                                       # EDA 결과물
│   │   ├── plots/                                 # EDA 시각화 이미지 저장
│   │   │   ├── .gitkeep                           # 빈 폴더 유지를 위한 파일
│   │   │   ├── bar_mean_by_churn_error_rate.png   # churn 여부별 error_rate 평균 막대그래프
│   │   │   ├── bar_mean_by_churn_health_score.png # churn 여부별 health_score 평균 막대그래프
│   │   │   ├── bar_mean_by_churn_satisfaction.png # churn 여부별 만족도 평균 막대그래프
│   │   │   ├── bar_mean_by_churn_usage.png        # churn 여부별 사용량 평균 막대그래프
│   │   │   ├── correlation_heatmap_key_features.png # 주요 변수 상관관계 히트맵
│   │   │   ├── dummy_group_summary_country.png    # 국가 더미 변수 요약 시각화
│   │   │   ├── dummy_group_summary_industry.png   # 산업 더미 변수 요약 시각화
│   │   │   ├── dummy_group_summary_referral.png   # 유입경로 더미 변수 요약 시각화
│   │   │   ├── hist_account_age_days.png          # account_age_days 분포 히스토그램
│   │   │   ├── hist_avg_mrr_amount.png            # 평균 MRR 분포 히스토그램
│   │   │   ├── hist_days_since_last_usage.png     # 최근 사용 이후 경과일 분포 히스토그램
│   │   │   ├── hist_health_score.png              # health_score 분포 히스토그램
│   │   │   ├── hist_total_subscriptions.png       # 총 구독 수 분포 히스토그램
│   │   │   ├── missing_pattern_heatmap.png        # 결측 패턴 히트맵
│   │   │   └── target_distribution_overall.png    # churn 타깃 분포 시각화
│   │   └── tables/                                # EDA 표 형태 결과 저장
│   │       ├── .gitkeep                           # 빈 폴더 유지를 위한 파일
│   │       ├── correlation_matrix_key_features.csv # 주요 변수 상관행렬
│   │       ├── correlation_with_churn.csv         # churn과 각 변수의 상관 요약
│   │       ├── dummy_group_summary_country.csv    # 국가 더미 변수 요약표
│   │       ├── dummy_group_summary_industry.csv   # 산업 더미 변수 요약표
│   │       ├── dummy_group_summary_referral.csv   # 유입경로 더미 변수 요약표
│   │       ├── eda_summary_report.csv             # EDA 전체 요약 리포트
│   │       ├── feature_importance_precheck.csv    # 사전 중요 변수 점검 결과
│   │       ├── group_mean_by_churn.csv            # churn 그룹별 평균 비교표
│   │       ├── missing_counts.csv                 # 변수별 결측 개수 표
│   │       ├── missing_ratio.csv                  # 변수별 결측 비율 표
│   │       ├── missing_vs_churn.csv               # 결측과 churn 관계 요약표
│   │       ├── numeric_summary.csv                # 수치형 변수 기초통계표
│   │       ├── skewness_summary.csv               # 왜도 요약표
│   │       └── target_distribution_summary.csv    # 타깃 분포 요약표
│   ├── models/                                    # 모델 학습 및 평가 결과물
│   │   ├── .gitkeep                               # 빈 폴더 유지를 위한 파일
│   │   ├── baseline_metrics.csv                   # 베이스라인 모델 성능 지표
│   │   ├── baseline_model.pkl                     # 베이스라인 모델 저장 파일
│   │   ├── best_model.pkl                         # 최종 선택 모델 저장 파일
│   │   ├── confusion_matrix.csv                   # 혼동행렬 수치 결과
│   │   ├── confusion_matrix.png                   # 혼동행렬 시각화 이미지
│   │   ├── confusion_matrix_tree.csv              # 트리 모델 혼동행렬 결과
│   │   ├── dl_feature_columns.csv                 # 딥러닝 입력 피처 목록
│   │   ├── dl_metrics.csv                         # 딥러닝 모델 성능 지표
│   │   ├── dl_model.pth                           # PyTorch 딥러닝 모델 가중치 파일
│   │   ├── dl_scaler.pkl                          # 딥러닝 입력 스케일러 저장 파일
│   │   ├── dl_test_predictions.csv                # 딥러닝 테스트 예측 결과
│   │   ├── feature_importance.csv                 # 변수 중요도 표
│   │   ├── feature_importance.png                 # 변수 중요도 그래프
│   │   ├── model_comparison.csv                   # 모델 간 성능 비교표
│   │   ├── model_comparison_tuned.csv             # threshold 조정 후 모델 비교표
│   │   ├── pr_curve.png                           # Precision-Recall 곡선 이미지
│   │   ├── pr_curve_points.csv                    # PR 곡선 좌표값
│   │   ├── pr_curve_points_tree.csv               # 트리 모델 PR 곡선 좌표값
│   │   ├── roc_curve.png                          # ROC 곡선 이미지
│   │   ├── roc_curve_points.csv                   # ROC 곡선 좌표값
│   │   ├── roc_curve_points_tree.csv              # 트리 모델 ROC 곡선 좌표값
│   │   ├── threshold_metrics.csv                  # threshold별 성능 지표
│   │   └── threshold_metrics_all_models.csv       # 모든 모델의 threshold 비교표
│   ├── streamlit/                                 # Streamlit 앱용 산출물
│   │   ├── .gitkeep                               # 빈 폴더 유지를 위한 파일
│   │   ├── dashboard_summary.csv                  # 대시보드 요약 데이터
│   │   └── sample_prediction.csv                  # 예측 예시용 샘플 데이터
│   └── xai/                                       # 설명가능 AI 결과물
│       ├── .gitkeep                               # 빈 폴더 유지를 위한 파일
│       ├── local_explanation_sample_1.csv         # 개별 고객 예측 설명 샘플 1
│       ├── local_explanation_sample_2.csv         # 개별 고객 예측 설명 샘플 2
│       ├── reason_mapping_report.csv              # 예측 근거를 해석 문장으로 매핑한 보고서
│       ├── shap_bar.png                           # SHAP 전역 중요도 막대그래프
│       ├── shap_dependence_error.png              # error 관련 변수 SHAP dependence plot
│       ├── shap_dependence_usage.png              # usage 관련 변수 SHAP dependence plot
│       ├── shap_summary.png                       # SHAP summary plot
│       └── xai_summary_report.csv                 # XAI 결과 요약표

├── src/                                           # 실제 프로젝트 소스코드
│   ├── __init__.py                                # src 패키지 초기화 파일
│
│   ├── app/                                       # Streamlit 앱 코드
│   │   ├── __init__.py                            # app 패키지 초기화 파일
│   │   ├── streamlit_app.py                       # Streamlit 메인 실행 파일
│   │   ├── sections/                              # 앱 화면별 섹션 코드
│   │   │   ├── eda_section.py                     # EDA 화면 렌더링 코드
│   │   │   ├── model_section.py                   # 모델 성능 화면 렌더링 코드
│   │   │   ├── overview_section.py                # 프로젝트 개요 화면 렌더링 코드
│   │   │   ├── prediction_section.py              # 고객별 예측 화면 렌더링 코드
│   │   │   └── xai_section.py                     # XAI 설명 화면 렌더링 코드
│   │   └── utils/                                 # 앱 보조 유틸 코드
│   │       ├── formatters.py                      # 숫자/문자 표시 형식 보조 함수
│   │       └── load_data.py                       # 앱에서 사용할 데이터 로딩 함수
│
│   ├── config/                                    # 경로·설정 관리 코드
│   │   ├── __init__.py                            # config 패키지 초기화 파일
│   │   ├── paths.py                               # 프로젝트 폴더 경로 정의
│   │   └── settings.py                            # 공통 설정값 관리
│
│   ├── data/                                      # 데이터 점검·전처리 코드
│   │   ├── __init__.py                            # data 패키지 초기화 파일
│   │   ├── data_check.py                          # 원천 데이터 품질 점검 스크립트
│   │   ├── make_analysis_table.py                 # 분석용 최종 테이블 생성 스크립트
│   │   ├── make_train_table.py                    # 학습용 기본 테이블 생성 스크립트
│   │   ├── preprocess_accounts.py                 # accounts 전처리 스크립트
│   │   ├── preprocess_churn_events.py             # churn_events 전처리 스크립트
│   │   ├── preprocess_feature_usage.py            # feature_usage 전처리/집계 스크립트
│   │   ├── preprocess_subscriptions.py            # subscriptions 전처리 스크립트
│   │   ├── preprocess_support_tickets.py          # support_tickets 전처리/집계 스크립트
│   │   └── split_dataset.py                       # train/valid/test 분리 스크립트
│
│   ├── eda/                                       # 탐색적 데이터 분석 코드
│   │   ├── __init__.py                            # eda 패키지 초기화 파일
│   │   ├── eda_by_churn.py                        # churn 기준 비교 분석 코드
│   │   ├── eda_categoricals.py                    # 범주형 변수 EDA 코드
│   │   ├── eda_main.py                            # EDA 전체 실행 메인 스크립트
│   │   ├── eda_missingness.py                     # 결측치 분석 코드
│   │   ├── eda_numeric.py                         # 수치형 변수 분석 코드
│   │   └── eda_visualization.py                   # EDA 시각화 생성 코드
│
│   ├── features/                                  # 피처 엔지니어링 코드
│   │   ├── __init__.py                            # features 패키지 초기화 파일
│   │   ├── build_features.py                      # 파생변수 생성 스크립트
│   │   ├── encode_categoricals.py                 # 범주형 변수 인코딩 스크립트
│   │   ├── make_dl_dataset.py                     # 딥러닝용 입력 데이터셋 생성 스크립트
│   │   ├── missing_flags.py                       # 결측 여부 flag 변수 생성 코드
│   │   └── subscription_change_features.py        # 구독 변화 관련 파생변수 생성 코드
│
│   ├── models/                                    # 모델 학습·예측·평가 코드
│   │   ├── __init__.py                            # models 패키지 초기화 파일
│   │   ├── compare_models.py                      # 여러 모델 성능 비교 코드
│   │   ├── evaluate.py                            # 성능 평가 함수 모음
│   │   ├── predict.py                             # 일반 모델 예측 스크립트
│   │   ├── predict_dl_model.py                    # 딥러닝 모델 예측 스크립트
│   │   ├── save_model.py                          # 모델 저장 보조 코드
│   │   ├── threshold_tuning.py                    # threshold 조정 로직
│   │   ├── train_baseline.py                      # 베이스라인 모델 학습 스크립트
│   │   ├── train_dl_model.py                      # 딥러닝 모델 학습 스크립트
│   │   ├── train_tree_model.py                    # 트리 기반 모델 학습 스크립트
│   │   └── tune_thresholds.py                     # 모델별 threshold 탐색 실행 스크립트
│
│   ├── processed/                                 # src 내부 보조 산출물 폴더
│   │   ├── 01_data_preprocessing.ipynb            # 전처리 관련 노트북
│   │   └── 01_preprocessed_data.csv               # 전처리 결과 예시 데이터
│
│   ├── utils/                                     # 공통 유틸리티 코드
│   │   ├── __init__.py                            # utils 패키지 초기화 파일
│   │   ├── helpers.py                             # 보조 함수 모음
│   │   ├── io.py                                  # 입출력 관련 유틸 함수
│   │   ├── logger.py                              # 로그 출력 설정 코드
│   │   └── plot_utils.py                          # 그래프 생성 보조 함수
│
│   └── xai/                                       # 설명가능 AI 분석 코드
│       ├── __init__.py                            # xai 패키지 초기화 파일
│       ├── global_explanations.py                 # 전역 설명 생성 코드
│       ├── local_explanations.py                  # 개별 예측 설명 생성 코드
│       ├── reason_mapping.py                      # SHAP 결과를 해석 문장으로 변환하는 코드
│       └── shap_analysis.py                       # SHAP 분석 메인 스크립트
```
