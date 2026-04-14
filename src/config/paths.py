# paths.py
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

ASSETS_DIR = ROOT_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
EDA_TABLES_DIR = EDA_DIR / "tables"
EDA_PLOTS_DIR = EDA_DIR / "plots"
MODELS_OUTPUT_DIR = OUTPUTS_DIR / "models"
XAI_OUTPUT_DIR = OUTPUTS_DIR / "xai"
STREAMLIT_OUTPUT_DIR = OUTPUTS_DIR / "streamlit"

DOCS_DIR = ROOT_DIR / "docs"

for directory in [
    ASSETS_DIR, IMAGES_DIR, DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    OUTPUTS_DIR, EDA_DIR, EDA_TABLES_DIR, EDA_PLOTS_DIR, MODELS_OUTPUT_DIR,
    XAI_OUTPUT_DIR, STREAMLIT_OUTPUT_DIR, DOCS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
