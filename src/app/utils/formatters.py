# formatters.py

from __future__ import annotations  # 타입 힌트에서 forward reference 사용 가능하게 설정


# 비율(0~1 값)을 퍼센트 문자열로 변환하는 함수
def format_pct(value: float) -> str:
    # value를 100배 후 소수점 2자리까지 표시하고 % 붙여 반환
    return f"{value * 100:.2f}%"


# 숫자를 천 단위 콤마가 포함된 문자열로 변환하는 함수
def format_int(value) -> str:
    try:
        # 정수로 변환 후 천 단위 콤마 포맷 적용
        return f"{int(value):,}"
    except Exception:
        # 숫자로 변환이 불가능한 경우 문자열 그대로 반환
        return str(value)