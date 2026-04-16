# paths.py

# paths.py

from pathlib import Path  # 파일 및 폴더 경로를 객체 형태로 다루기 위한 모듈


# 현재 파일 기준으로 프로젝트 루트 경로 설정 (src/config/paths.py 기준으로 2단계 위)
ROOT_DIR = Path(__file__).resolve().parents[2]


# -------------------------------
# assets 관련 경로
# -------------------------------

ASSETS_DIR = ROOT_DIR / "assets"          # 정적 파일(이미지, 아이콘 등) 저장 폴더
IMAGES_DIR = ASSETS_DIR / "images"        # 이미지 파일 전용 폴더


# -------------------------------
# 데이터 관련 경로
# -------------------------------

DATA_DIR = ROOT_DIR / "data"              # 전체 데이터 폴더
RAW_DIR = DATA_DIR / "raw"                # 원천 데이터 (가공 전)
INTERIM_DIR = DATA_DIR / "interim"        # 중간 처리 데이터 (전처리 중간 단계)
PROCESSED_DIR = DATA_DIR / "processed"    # 최종 분석/모델 입력용 데이터


# -------------------------------
# 출력 결과 관련 경로
# -------------------------------

OUTPUTS_DIR = ROOT_DIR / "outputs"        # 모든 결과물 저장 루트 폴더

EDA_DIR = OUTPUTS_DIR / "eda"             # EDA 결과 폴더
EDA_TABLES_DIR = EDA_DIR / "tables"       # EDA 결과 테이블
EDA_PLOTS_DIR = EDA_DIR / "plots"         # EDA 시각화 결과

MODELS_OUTPUT_DIR = OUTPUTS_DIR / "models"  # 모델 학습 결과 및 저장 파일
XAI_OUTPUT_DIR = OUTPUTS_DIR / "xai"        # SHAP 및 XAI 결과
STREAMLIT_OUTPUT_DIR = OUTPUTS_DIR / "streamlit"  # Streamlit용 가공 데이터


# -------------------------------
# 문서 관련 경로
# -------------------------------

DOCS_DIR = ROOT_DIR / "docs"              # README, 기술서 등 문서 저장 폴더


# -------------------------------
# 폴더 자동 생성
# -------------------------------
# 위에서 정의한 모든 디렉토리가 실제로 존재하도록 생성 (없으면 자동 생성)
for directory in [
    ASSETS_DIR, IMAGES_DIR, DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    OUTPUTS_DIR, EDA_DIR, EDA_TABLES_DIR, EDA_PLOTS_DIR, MODELS_OUTPUT_DIR,
    XAI_OUTPUT_DIR, STREAMLIT_OUTPUT_DIR, DOCS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)  # 상위 폴더까지 포함해 생성, 이미 있으면 무시