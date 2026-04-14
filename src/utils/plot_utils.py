# plot_utils.py
from __future__ import annotations

from pathlib import Path
import platform
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def _pick_font() -> str:
    system_name = platform.system()
    installed = {f.name for f in fm.fontManager.ttflist}

    candidates_by_os = {
        "Windows": ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"],
        "Darwin": ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"],
        "Linux": ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"],
    }
    candidates = candidates_by_os.get(system_name, ["DejaVu Sans"])
    for candidate in candidates:
        if candidate in installed:
            return candidate
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
    font_name = _pick_font()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    return font_name


def apply_plot_style() -> str:
    font_name = set_korean_font()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 120
    plt.rcParams["axes.grid"] = False
    return font_name


def save_figure(output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
