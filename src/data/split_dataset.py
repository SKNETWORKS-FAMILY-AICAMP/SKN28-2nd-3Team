from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.paths import PROCESSED_DIR
from src.config.settings import RANDOM_STATE, TARGET_COL, VALID_SIZE, TEST_SIZE
from src.utils.io import read_csv, save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_dataset() -> None:
    df = read_csv(PROCESSED_DIR / "train_table_ml.csv")
    y = df[TARGET_COL]
    id_cols = [c for c in ["account_id"] if c in df.columns]
    X = df.drop(columns=[TARGET_COL])

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    valid_ratio_adjusted = VALID_SIZE / (1 - TEST_SIZE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=valid_ratio_adjusted,
        random_state=RANDOM_STATE, stratify=y_train_valid
    )

    train = X_train.copy(); train[TARGET_COL] = y_train.values
    valid = X_valid.copy(); valid[TARGET_COL] = y_valid.values
    test = X_test.copy(); test[TARGET_COL] = y_test.values

    save_csv(train, PROCESSED_DIR / "train.csv")
    save_csv(valid, PROCESSED_DIR / "valid.csv")
    save_csv(test, PROCESSED_DIR / "test.csv")
    save_csv(X_train, PROCESSED_DIR / "X_train.csv")
    save_csv(X_valid, PROCESSED_DIR / "X_valid.csv")
    save_csv(X_test, PROCESSED_DIR / "X_test.csv")
    save_csv(y_train.to_frame(name=TARGET_COL), PROCESSED_DIR / "y_train.csv")
    save_csv(y_valid.to_frame(name=TARGET_COL), PROCESSED_DIR / "y_valid.csv")
    save_csv(y_test.to_frame(name=TARGET_COL), PROCESSED_DIR / "y_test.csv")
    logger.info("saved train/valid/test splits")


def main() -> None:
    split_dataset()


if __name__ == "__main__":
    main()
