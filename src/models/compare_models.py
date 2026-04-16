# compare_models.py

"""
이 파일은 여러 모델의 성능 평가 결과를 하나의 비교표로 통합하여
최종 모델 간 성능을 비교할 수 있도록 정리하는 모듈이다.

주요 역할:
- 저장된 모델 성능 CSV 파일 존재 여부 확인 및 로드
- 모델명 컬럼명을 일관된 형식으로 표준화
- 비교에 필요한 핵심 평가지표만 선택
- baseline 모델과 DL 모델 성능 결과를 하나로 병합
- 중복 모델 제거 및 roc_auc 기준 정렬
- 최종 모델 비교표를 CSV로 저장하고 출력
"""
from pathlib import Path  # 파일 및 폴더 경로 처리를 위한 객체
import pandas as pd  # 데이터프레임 처리 라이브러리


# 현재 파일 위치를 기준으로 프로젝트 루트 경로 계산
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 모델 결과물이 저장되는 폴더 경로 설정
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    # 파일이 존재하면 CSV를 읽어 DataFrame으로 반환
    if path.exists():
        return pd.read_csv(path)

    # 파일이 없으면 None 반환
    return None


def standardize_model_name(df: pd.DataFrame) -> pd.DataFrame:
    # 원본 데이터 손상을 막기 위해 복사본 생성
    df = df.copy()

    # model_name 컬럼이 있고 model 컬럼이 없으면 model로 이름 통일
    if "model_name" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"model_name": "model"})

    # Model 컬럼이 있고 model 컬럼이 없으면 model로 이름 통일
    if "Model" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"Model": "model"})

    # 표준화된 DataFrame 반환
    return df


def keep_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 최종 비교표에 유지할 핵심 성능 지표 컬럼 목록
    wanted = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]

    # 실제 존재하는 컬럼만 선택
    existing = [col for col in wanted if col in df.columns]

    # 선택된 컬럼만 남긴 DataFrame 반환
    return df[existing].copy()


def main():
    # 비교 대상 파일 경로 설정
    comparison_path = MODEL_DIR / "model_comparison.csv"
    baseline_path = MODEL_DIR / "baseline_metrics.csv"
    dl_path = MODEL_DIR / "dl_metrics.csv"

    # 병합할 DataFrame들을 담을 리스트
    frames = []

    # -----------------------------------
    # 1. 기존 model_comparison.csv 불러오기
    # -----------------------------------
    # 이미 비교표가 있으면 우선 로드해서 재사용
    existing_comparison = load_csv_if_exists(comparison_path)
    if existing_comparison is not None and not existing_comparison.empty:
        existing_comparison = standardize_model_name(existing_comparison)  # 모델명 컬럼 표준화
        existing_comparison = keep_metric_columns(existing_comparison)  # 핵심 지표만 유지
        frames.append(existing_comparison)

    # -----------------------------------
    # 2. baseline_metrics.csv 불러오기
    # -----------------------------------
    # 전통적인 ML baseline 모델 성능 결과 추가
    baseline_df = load_csv_if_exists(baseline_path)
    if baseline_df is not None and not baseline_df.empty:
        baseline_df = standardize_model_name(baseline_df)  # 모델명 컬럼 표준화
        baseline_df = keep_metric_columns(baseline_df)  # 핵심 지표만 유지
        frames.append(baseline_df)

    # -----------------------------------
    # 3. dl_metrics.csv 불러오기
    # -----------------------------------
    # 딥러닝 모델 성능 결과는 반드시 있어야 하므로 없으면 에러 발생
    dl_df = load_csv_if_exists(dl_path)
    if dl_df is None or dl_df.empty:
        raise FileNotFoundError(
            "dl_metrics.csv가 없습니다.\n"
            "먼저 python -m src.models.predict_dl_model 을 실행하세요."
        )

    # DL 결과도 동일하게 정리 후 추가
    dl_df = standardize_model_name(dl_df)
    dl_df = keep_metric_columns(dl_df)
    frames.append(dl_df)

    # -----------------------------------
    # 4. 모든 결과 병합
    # -----------------------------------
    merged = pd.concat(frames, ignore_index=True)

    # -----------------------------------
    # 5. 중복 모델 제거
    # -----------------------------------
    # 같은 모델명이 중복되면 뒤에 온 결과를 우선 유지
    if "model" in merged.columns:
        merged = merged.drop_duplicates(subset=["model"], keep="last")

    # -----------------------------------
    # 6. 성능 기준 정렬
    # -----------------------------------
    # roc_auc 기준 내림차순 정렬 (클수록 좋은 모델)
    if "roc_auc" in merged.columns:
        merged = merged.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)

    # -----------------------------------
    # 7. 최종 비교표 저장
    # -----------------------------------
    merged.to_csv(comparison_path, index=False)

    # 저장 완료 메시지 및 결과 출력
    print(f"최종 모델 비교표 저장 완료: {comparison_path}")
    print(merged)


# 스크립트를 직접 실행할 때 main 함수 호출
if __name__ == "__main__":
    main()