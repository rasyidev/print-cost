"""
Configuration module for Print Cost application.

This module centralizes all configuration values to eliminate hardcoded constants
throughout the codebase and enable environment-based configuration.
"""

import os
from pathlib import Path
from typing import Dict

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Model configuration
MODEL_DIR = ROOT_DIR / "models"
DEFAULT_MODEL_FILE = "xgboost_98.64_cmy_k_cmyk_7_dpi.pkl"
DEFAULT_MODEL_PATH = MODEL_DIR / DEFAULT_MODEL_FILE
PRICE_LABEL_FILE = MODEL_DIR / "price_label.json"

# Model metadata
MODEL_FEATURES = ["cmy", "k", "cmyk"]
MODEL_VERSION = "1.0.0"
MODEL_F1_SCORE = 0.99

# PDF Processing configuration
DEFAULT_DPI = 7  # DPI for PDF rendering
SUPPORTED_FILE_EXTENSIONS = ["pdf"]
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB

# Price mappings
PRICE_LABEL_MAP: Dict[int, int] = {
    0: 500,   # Mono Print
    1: 750,   # Color Light
    2: 1000,  # Color Standard
    3: 1500,  # Color Heavy
    4: 2000,  # Full Color – Dark & Mixed
}

PRICE_CATEGORY_MAP: Dict[int, str] = {
    500: "Mono Print",
    750: "Color Light",
    1000: "Color Standard",
    1500: "Color Heavy",
    2000: "Full Color – Dark & Mixed",
}

# Color mappings for visualization
CATEGORY_COLOR_MAP: Dict[str, str] = {
    "Mono Print": "#FFF2EF",
    "Color Light": "#FFDBB6",
    "Color Standard": "#F7A5A5",
    "Color Heavy": "#5D688A",
    "Full Color – Dark & Mixed": "#88527F",
}

# Validation thresholds
MIN_PAGES = 1
MAX_PAGES = 1000
VALID_CMYK_RANGE = (0.0, 100.0)

# A4 paper dimensions in points (1 point = 1/72 inch)
A4_WIDTH_PT = 595
A4_HEIGHT_PT = 842
PAGE_SIZE_TOLERANCE = 10  # points tolerance for A4 validation

# Environment variables (for production deployment)
def get_env_config() -> Dict[str, str]:
    """
    Get environment-specific configuration.
    
    Returns:
        Dict containing environment configuration
    """
    return {
        "MODEL_PATH": os.getenv("PRINT_COST_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        "MAX_FILE_SIZE_MB": int(os.getenv("PRINT_COST_MAX_FILE_SIZE", MAX_FILE_SIZE_MB)),
        "DPI": int(os.getenv("PRINT_COST_DPI", DEFAULT_DPI)),
    }
