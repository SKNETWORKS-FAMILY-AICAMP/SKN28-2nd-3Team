# io.py

"""
이 파일은 CSV 파일을 읽고 저장하는 공통 입출력 유틸 모듈이다.

주요 역할:
- 지정한 경로의 CSV 파일을 DataFrame으로 읽기
- 저장 경로의 상위 폴더가 없으면 자동 생성
- DataFrame을 utf-8-sig 인코딩으로 CSV 파일로 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

from pathlib import Path  # 파일 및 폴더 경로 처리를 위한 객체
import pandas as pd  # 데이터프레임 처리 라이브러리


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    # 지정한 경로의 CSV 파일을 읽어 DataFrame으로 반환
    # **kwargs를 통해 parse_dates, usecols 등 추가 옵션도 함께 전달 가능
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    # 저장할 경로의 상위 폴더가 없으면 자동 생성
    path.parent.mkdir(parents=True, exist_ok=True)

    # DataFrame을 CSV 파일로 저장
    # encoding="utf-8-sig"는 한글이 포함된 CSV를 엑셀에서 열 때 깨짐을 줄이기 위함
    df.to_csv(path, index=index, encoding="utf-8-sig")