"""
config.py
---------
Central configuration for the Fintech Risk Terrain project.
All dataset codes, filter parameters, file paths, and feature
definitions live here. Edit this file to adjust scope or years.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & data directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Create directories on import so scripts never fail on missing folders
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Country & year scope
# ---------------------------------------------------------------------------
TARGET_COUNTRY = "HU"           # Hungary — ISO 3166-1 alpha-2
COMPARISON_COUNTRIES = [        # CEE peers for benchmarking
    "HU", "PL", "CZ", "SK", "RO", "AT",
]
YEAR_START = 2015
YEAR_END   = 2023               # Latest available across all datasets

# ---------------------------------------------------------------------------
# Eurostat dataset registry
# Each entry: (dataset_code, human_label, key_dimensions_to_keep)
# ---------------------------------------------------------------------------
EUROSTAT_DATASETS = {

    # 1. Household budget structure — share of spending by category
    "hbs_str_t223": {
        "label": "Household budget structure",
        "filters": {"geo": COMPARISON_COUNTRIES},
        "keep_cols": ["geo", "time", "coicop", "values"],
        "coicop_of_interest": [
            "CP01",   # Food and non-alcoholic beverages
            "CP04",   # Housing, water, electricity, gas
            "CP12",   # Miscellaneous goods and services (proxy discretionary)
        ],
    },

    # 2. Inability to face unexpected financial expenses
    "ilc_mdes09": {
        "label": "Unable to meet unexpected expense",
        "filters": {"geo": COMPARISON_COUNTRIES},
        "keep_cols": ["geo", "time", "quantile", "values"],
    },

    # 3. Material deprivation by income quintile
    "ilc_lvhl11": {
        "label": "Material deprivation rate",
        "filters": {"geo": COMPARISON_COUNTRIES},
        "keep_cols": ["geo", "time", "quantile", "values"],
    },

    # 4. Income quintile share ratio (S80/S20 inequality measure)
    "ilc_di12": {
        "label": "Income quintile share ratio",
        "filters": {"geo": COMPARISON_COUNTRIES},
        "keep_cols": ["geo", "time", "values"],
    },

    # 5. Housing cost overburden rate
    "ilc_lvho07a": {
        "label": "Housing cost overburden",
        "filters": {"geo": COMPARISON_COUNTRIES},
        "keep_cols": ["geo", "time", "incgrp", "tenure", "values"],
    },

    # 6. Arrears on mortgage or rent
    "ilc_lvho05a": {
        "label": "Arrears on mortgage or rent",
        "filters": {"geo": COMPARISON_COUNTRIES},
        "keep_cols": ["geo", "time", "incgrp", "values"],
    },

    # 7. Wage distribution reference (for income anchoring)
    "earn_ses_pub2s": {
        "label": "Earnings by sex and economic activity",
        "filters": {"geo": COMPARISON_COUNTRIES},
        "keep_cols": ["geo", "time", "nace_r2", "values"],
    },
}

# ---------------------------------------------------------------------------
# MNB (Magyar Nemzeti Bank) PDF data
# Financial Stability Report — household debt section
# ---------------------------------------------------------------------------
MNB_PDF_URL = (
    "https://www.mnb.hu/letoltes/financial-stability-report-2023h2.pdf"
)
MNB_PDF_LOCAL = DATA_RAW / "mnb_financial_stability_2023h2.pdf"

# Page range known to contain household debt-by-decile tables
# (inspect PDF manually and update if MNB changes layout)
MNB_TABLE_PAGES = list(range(20, 45))   # pages 20–44 (0-indexed in pdfplumber)

# Column names we expect to extract from the MNB tables
MNB_EXPECTED_COLS = [
    "income_decile",
    "debt_to_income_ratio",
    "debt_service_ratio",
    "share_of_indebted_households",
    "npl_rate",
]

# ---------------------------------------------------------------------------
# Feature definitions (used in 03_merge_features.py)
# ---------------------------------------------------------------------------
FEATURES = [
    "dti_proxy",             # Debt-to-income ratio (from MNB)
    "housing_burden",        # Housing cost as % of income (ilc_lvho07a)
    "buffer_score",          # Inverse of inability to meet unexpected expense
    "deprivation_index",     # Material deprivation composite (ilc_lvhl11)
    "consumption_stress",    # Food + utilities share of total spend (hbs_str_t223)
    "arrears_exposure",      # Arrears probability (ilc_lvho05a)
    "income_volatility",     # Wage CV proxy (earn_ses_pub2s)
]

# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------
OUTPUT_EUROSTAT_RAW    = DATA_RAW / "eurostat_raw.pkl"
OUTPUT_MNB_RAW         = DATA_RAW / "mnb_raw.csv"
OUTPUT_FEATURE_MATRIX  = DATA_PROCESSED / "feature_matrix.csv"
OUTPUT_SCALED_MATRIX   = DATA_PROCESSED / "feature_matrix_scaled.csv"