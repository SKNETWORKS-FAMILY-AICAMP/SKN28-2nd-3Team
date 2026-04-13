from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def standardize_model_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "model_name" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"model_name": "model"})

    if "Model" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"Model": "model"})

    return df


def keep_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    existing = [col for col in wanted if col in df.columns]
    return df[existing].copy()


def main():
    comparison_path = MODEL_DIR / "model_comparison.csv"
    baseline_path = MODEL_DIR / "baseline_metrics.csv"
    dl_path = MODEL_DIR / "dl_metrics.csv"

    frames = []

    # 기존 model_comparison.csv가 있으면 우선 사용
    existing_comparison = load_csv_if_exists(comparison_path)
    if existing_comparison is not None and not existing_comparison.empty:
        existing_comparison = standardize_model_name(existing_comparison)
        existing_comparison = keep_metric_columns(existing_comparison)
        frames.append(existing_comparison)

    # baseline_metrics.csv도 별도로 합치기
    baseline_df = load_csv_if_exists(baseline_path)
    if baseline_df is not None and not baseline_df.empty:
        baseline_df = standardize_model_name(baseline_df)
        baseline_df = keep_metric_columns(baseline_df)
        frames.append(baseline_df)

    # DL metrics 합치기
    dl_df = load_csv_if_exists(dl_path)
    if dl_df is None or dl_df.empty:
        raise FileNotFoundError(
            "dl_metrics.csv가 없습니다.\n"
            "먼저 python -m src.models.predict_dl_model 을 실행하세요."
        )

    dl_df = standardize_model_name(dl_df)
    dl_df = keep_metric_columns(dl_df)
    frames.append(dl_df)

    merged = pd.concat(frames, ignore_index=True)

    # 중복 model 제거: 뒤에 온 값 우선
    if "model" in merged.columns:
        merged = merged.drop_duplicates(subset=["model"], keep="last")

    # roc_auc 기준 정렬
    if "roc_auc" in merged.columns:
        merged = merged.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)

    merged.to_csv(comparison_path, index=False)

    print(f"최종 모델 비교표 저장 완료: {comparison_path}")
    print(merged)


if __name__ == "__main__":
    main()