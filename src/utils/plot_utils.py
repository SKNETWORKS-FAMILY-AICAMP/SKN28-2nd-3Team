# plot_utils.py

"""
이 파일은 matplotlib 시각화에서 한글 폰트와 공통 그래프 스타일을 안정적으로 적용하고,
그래프 이미지를 파일로 저장하는 유틸 모듈이다.

주요 역할:
- 운영체제별 사용 가능한 한글 폰트 자동 탐색
- matplotlib 한글 폰트 및 마이너스 깨짐 설정
- 공통 figure 크기, dpi, grid 등 시각화 스타일 적용
- 저장 경로가 없으면 폴더를 자동 생성한 뒤 figure 저장
"""
from __future__ import annotations  # 타입 힌트 지연 평가 (forward reference 지원)

from pathlib import Path  # 파일 및 폴더 경로 처리를 위한 객체
import platform  # 현재 운영체제 이름 확인용 모듈
import warnings  # 불필요한 경고 메시지 제어용 모듈

import matplotlib.pyplot as plt  # 시각화 라이브러리
from matplotlib import font_manager as fm  # 설치된 폰트 목록 확인용 모듈


def _pick_font() -> str:
    # 현재 운영체제 이름 확인 (예: Windows, Darwin, Linux)
    system_name = platform.system()

    # 현재 matplotlib가 인식하는 설치 폰트 이름 목록 수집
    installed = {f.name for f in fm.fontManager.ttflist}

    # 운영체제별 우선 사용 후보 폰트 정의
    candidates_by_os = {
        "Windows": ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"],
        "Darwin": ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"],
        "Linux": ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"],
    }

    # 현재 운영체제에 맞는 폰트 후보 목록 가져오기
    # 정의되지 않은 OS라면 DejaVu Sans를 기본값으로 사용
    candidates = candidates_by_os.get(system_name, ["DejaVu Sans"])

    # 후보 폰트 중 실제 설치된 폰트를 순서대로 탐색
    for candidate in candidates:
        if candidate in installed:
            return candidate

    # 후보 중 아무것도 없으면 최종 fallback 폰트 반환
    return "DejaVu Sans"


def set_korean_font() -> str:
    """
    운영체제별 한글 폰트 자동 설정
    - Windows 우선: Malgun Gothic
    - macOS 우선: AppleGothic
    - Linux 우선: Nanum/Noto 계열
    없으면 DejaVu Sans로 fallback

    macOS 참고:
    - AppleGothic 기본 사용
    - 별도 한글 폰트 설치 시 NanumGothic 등으로 자동 감지 가능
    - 설치 후 반영이 안 되면 matplotlib 캐시 삭제:
      rm -rf ~/.matplotlib
      rm -rf ~/Library/Caches/matplotlib
    """
    # 사용 가능한 한글 폰트 자동 선택
    font_name = _pick_font()

    # matplotlib 기본 폰트를 선택한 폰트로 설정
    plt.rcParams["font.family"] = font_name

    # 마이너스 기호가 깨지지 않도록 설정
    plt.rcParams["axes.unicode_minus"] = False

    # 폰트에 일부 glyph가 없을 때 뜨는 경고 메시지 숨김
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    # 실제 적용된 폰트 이름 반환
    return font_name


def apply_plot_style() -> str:
    # 한글 폰트 설정 적용
    font_name = set_korean_font()

    # 공통 figure 크기 설정
    plt.rcParams["figure.figsize"] = (10, 6)

    # 화면 출력 해상도 설정
    plt.rcParams["figure.dpi"] = 120

    # 저장 이미지 해상도 설정
    plt.rcParams["savefig.dpi"] = 120

    # 기본 grid 표시 비활성화
    plt.rcParams["axes.grid"] = False

    # 적용된 폰트 이름 반환
    return font_name


def save_figure(output_path: Path) -> None:
    # 전달받은 경로를 Path 객체로 변환
    output_path = Path(output_path)

    # 저장할 폴더가 없으면 자동 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 현재 figure를 지정한 경로에 저장
    plt.savefig(output_path, bbox_inches="tight")