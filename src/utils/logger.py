# logger.py

"""
이 파일은 프로젝트 전반에서 사용할 공통 logger를 생성하는 유틸 모듈이다.

주요 역할:
- 모듈별로 이름(name)을 가진 logger 생성
- 로그 레벨(INFO) 설정
- 콘솔 출력용 StreamHandler 추가
- 일관된 로그 포맷 적용
- 이미 handler가 있는 경우 중복 생성 방지
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

import logging  # 파이썬 기본 로깅 라이브러리


def get_logger(name: str) -> logging.Logger:
    # 전달받은 이름으로 logger 객체 생성 (또는 기존 logger 반환)
    logger = logging.getLogger(name)

    # 이미 handler가 존재하면 (중복 설정 방지)
    if logger.handlers:
        return logger

    # 로그 레벨을 INFO로 설정 (INFO 이상만 출력)
    logger.setLevel(logging.INFO)

    # 콘솔 출력용 handler 생성
    handler = logging.StreamHandler()

    # 로그 출력 형식 설정
    formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")

    # handler에 formatter 적용
    handler.setFormatter(formatter)

    # logger에 handler 추가
    logger.addHandler(handler)

    # 설정 완료된 logger 반환
    return logger