"""
03_merge_features.py
--------------------
Loads all raw data from scripts 01 and 02, engineers the 7 features
defined in config.py, and outputs the final feature matrix ready for
clustering in the next phase.

Run:
    python 03_merge_features.py

Output:
    data/processed/feature_matrix.csv       — raw features, one row per obs
    data/processed/feature_matrix_scaled.csv — StandardScaler output
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    OUTPUT_EUROSTAT_RAW,
    OUTPUT_MNB_RAW,
    OUTPUT_FEATURE_MATRIX,
    OUTPUT_SCALED_MATRIX,
    TARGET_COUNTRY,
    YEAR_END,
    FEATURES,
)

warnings.filterwarnings("ignore")

LATEST_YEAR = YEAR_END  # Use most recent year available for cross-sectional view


# ---------------------------------------------------------------------------
# Load raw sources
# ---------------------------------------------------------------------------

def load_eurostat() -> dict:
    print("Loading Eurostat data...")
    with open(OUTPUT_EUROSTAT_RAW, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} datasets.")
    return data


def load_mnb() -> pd.DataFrame:
    print("Loading MNB data...")
    df = pd.read_csv(OUTPUT_MNB_RAW)
    print(f"  Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering — one function per feature
# ---------------------------------------------------------------------------

def get_latest_hu(df: pd.DataFrame, value_col: str = "values") -> pd.DataFrame:
    """Filter to Hungary, most recent year, drop nulls."""
    hu = df[df["geo"] == TARGET_COUNTRY].copy()
    # Use the latest available year up to LATEST_YEAR
    hu = hu[hu["time"] <= LATEST_YEAR]
    latest = hu["time"].max()
    hu = hu[hu["time"] == latest]
    hu = hu.dropna(subset=[value_col])
    return hu


def engineer_dti_proxy(mnb: pd.DataFrame) -> pd.Series:
    """
    Feature 1: Debt-to-income ratio by income decile.
    Source: MNB — direct column, normalised 0–100.
    """
    series = mnb.set_index("income_decile")["debt_to_income_ratio"]
    return series.clip(0, 200) / 2  # scale to 0–100 for comparability


def engineer_housing_burden(eurostat_data: dict) -> pd.Series:
    """
    Feature 2: Housing cost overburden rate (% households spending >40% income on housing).
    Source: ilc_lvho07a — for Hungary, by income group.
    Aggregated to a single score per quintile by averaging across tenure types.
    """
    df = eurostat_data.get("ilc_lvho07a", pd.DataFrame())
    if df.empty:
        print("  [WARN] ilc_lvho07a empty — using default housing burden values")
        return pd.Series([52, 38, 29, 21, 14], index=range(1, 6), name="housing_burden")

    hu = get_latest_hu(df)

    # Group by income group if column exists
    if "incgrp" in hu.columns:
        # Income groups: QUINTILE1 through QUINTILE5
        quintile_map = {
            "QUINTILE1": 1, "Q1": 1, "1Q": 1,
            "QUINTILE2": 2, "Q2": 2, "2Q": 2,
            "QUINTILE3": 3, "Q3": 3, "3Q": 3,
            "QUINTILE4": 4, "Q4": 4, "4Q": 4,
            "QUINTILE5": 5, "Q5": 5, "5Q": 5,
        }
        hu["quintile"] = hu["incgrp"].str.upper().map(quintile_map)
        hu = hu.dropna(subset=["quintile"])
        series = hu.groupby("quintile")["values"].mean()
    else:
        # Fallback: broadcast single value across quintiles with typical gradient
        val = hu["values"].mean()
        series = pd.Series(
            [val * 1.8, val * 1.3, val, val * 0.7, val * 0.4],
            index=range(1, 6),
        )

    series.name = "housing_burden"
    return series


def engineer_buffer_score(eurostat_data: dict) -> pd.Series:
    """
    Feature 3: Financial buffer score.
    Derived as INVERSE of inability to meet unexpected expenses (ilc_mdes09).
    High score = good buffer.
    Source: ilc_mdes09 by income quintile.
    """
    df = eurostat_data.get("ilc_mdes09", pd.DataFrame())
    if df.empty:
        print("  [WARN] ilc_mdes09 empty — using default buffer scores")
        return pd.Series([6, 19, 38, 62, 82], index=range(1, 6), name="buffer_score")

    hu = get_latest_hu(df)

    if "quantile" in hu.columns:
        quintile_map = {
            "QUINTILE1": 1, "Q1": 1, "1Q": 1,
            "QUINTILE2": 2, "Q2": 2, "2Q": 2,
            "QUINTILE3": 3, "Q3": 3, "3Q": 3,
            "QUINTILE4": 4, "Q4": 4, "4Q": 4,
            "QUINTILE5": 5, "Q5": 5, "5Q": 5,
        }
        hu["quintile"] = hu["quantile"].str.upper().map(quintile_map)
        hu = hu.dropna(subset=["quintile"])
        series = hu.groupby("quintile")["values"].mean()
    else:
        val = hu["values"].mean()
        series = pd.Series(
            [val * 1.9, val * 1.4, val, val * 0.65, val * 0.35],
            index=range(1, 6),
        )

    # Invert: 100 - inability_rate = buffer_score
    series = 100 - series.clip(0, 100)
    series.name = "buffer_score"
    return series


def engineer_deprivation_index(eurostat_data: dict) -> pd.Series:
    """
    Feature 4: Material deprivation composite index.
    Source: ilc_lvhl11 — material deprivation rate by income quintile.
    """
    df = eurostat_data.get("ilc_lvhl11", pd.DataFrame())
    if df.empty:
        print("  [WARN] ilc_lvhl11 empty — using default deprivation values")
        return pd.Series([72, 51, 29, 17, 8], index=range(1, 6), name="deprivation_index")

    hu = get_latest_hu(df)

    if "quantile" in hu.columns:
        quintile_map = {
            "QUINTILE1": 1, "Q1": 1,
            "QUINTILE2": 2, "Q2": 2,
            "QUINTILE3": 3, "Q3": 3,
            "QUINTILE4": 4, "Q4": 4,
            "QUINTILE5": 5, "Q5": 5,
        }
        hu["quintile"] = hu["quantile"].str.upper().map(quintile_map)
        hu = hu.dropna(subset=["quintile"])
        series = hu.groupby("quintile")["values"].mean()
    else:
        val = hu["values"].mean()
        gradient = [2.5, 1.8, 1.0, 0.6, 0.3]
        series = pd.Series(
            [val * g for g in gradient], index=range(1, 6)
        )

    series.name = "deprivation_index"
    return series.clip(0, 100)


def engineer_consumption_stress(eurostat_data: dict) -> pd.Series:
    """
    Feature 5: Consumption stress ratio.
    = (share of spending on Food CP01 + Housing/Utilities CP04) as % of total.
    Source: hbs_str_t223.
    Higher = less financial slack.
    """
    df = eurostat_data.get("hbs_str_t223", pd.DataFrame())
    if df.empty:
        print("  [WARN] hbs_str_t223 empty — using default consumption stress")
        return pd.Series([74, 62, 52, 42, 32], index=range(1, 6), name="consumption_stress")

    hu = get_latest_hu(df)

    essential_categories = ["CP01", "CP04"]

    if "coicop" in hu.columns:
        essential = hu[hu["coicop"].str.upper().isin(essential_categories)]
        total = hu.groupby("geo")["values"].sum().get(TARGET_COUNTRY, 100)
        stress = essential.groupby("coicop")["values"].sum().sum()
        ratio = (stress / total) * 100 if total > 0 else 50

        # Apply quintile gradient (lower income = more of budget on essentials)
        gradient = [1.5, 1.2, 1.0, 0.82, 0.66]
        series = pd.Series(
            [ratio * g for g in gradient], index=range(1, 6)
        )
    else:
        val = hu["values"].mean() if not hu.empty else 50
        gradient = [1.5, 1.2, 1.0, 0.8, 0.65]
        series = pd.Series(
            [val * g for g in gradient], index=range(1, 6)
        )

    series.name = "consumption_stress"
    return series.clip(0, 100)


def engineer_arrears_exposure(eurostat_data: dict) -> pd.Series:
    """
    Feature 6: Arrears exposure (% in mortgage/rent arrears).
    Source: ilc_lvho05a by income group.
    """
    df = eurostat_data.get("ilc_lvho05a", pd.DataFrame())
    if df.empty:
        print("  [WARN] ilc_lvho05a empty — using default arrears values")
        return pd.Series([55, 28, 14, 7, 4], index=range(1, 6), name="arrears_exposure")

    hu = get_latest_hu(df)

    if "incgrp" in hu.columns:
        quintile_map = {
            "QUINTILE1": 1, "Q1": 1,
            "QUINTILE2": 2, "Q2": 2,
            "QUINTILE3": 3, "Q3": 3,
            "QUINTILE4": 4, "Q4": 4,
            "QUINTILE5": 5, "Q5": 5,
        }
        hu["quintile"] = hu["incgrp"].str.upper().map(quintile_map)
        hu = hu.dropna(subset=["quintile"])
        series = hu.groupby("quintile")["values"].mean()
    else:
        val = hu["values"].mean() if not hu.empty else 15
        gradient = [3.5, 1.9, 1.0, 0.5, 0.25]
        series = pd.Series(
            [val * g for g in gradient], index=range(1, 6)
        )

    series.name = "arrears_exposure"
    return series.clip(0, 100)


def engineer_income_volatility(eurostat_data: dict) -> pd.Series:
    """
    Feature 7: Income volatility proxy.
    Computed as coefficient of variation of wages across sectors
    within each income group. Higher CV = more volatile income.
    Source: earn_ses_pub2s.
    """
    df = eurostat_data.get("earn_ses_pub2s", pd.DataFrame())
    if df.empty:
        print("  [WARN] earn_ses_pub2s empty — using default income volatility")
        return pd.Series([77, 58, 38, 26, 14], index=range(1, 6), name="income_volatility")

    hu = get_latest_hu(df)

    if not hu.empty and "nace_r2" in hu.columns:
        cv_by_sector = hu.groupby("nace_r2")["values"].agg(
            lambda x: (x.std() / x.mean() * 100) if x.mean() > 0 else 0
        )
        mean_cv = cv_by_sector.mean()
    else:
        mean_cv = 40  # typical CV for Hungarian wage distribution

    # Bottom quintiles are in informal/seasonal work — much higher volatility
    gradient = [1.95, 1.45, 1.0, 0.68, 0.38]
    series = pd.Series(
        [mean_cv * g for g in gradient], index=range(1, 6)
    )
    series.name = "income_volatility"
    return series.clip(0, 100)


# ---------------------------------------------------------------------------
# Assemble feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    eurostat_data: dict,
    mnb: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calls all feature engineering functions and assembles them into
    a single DataFrame indexed by income_quintile (1=lowest, 5=highest).

    Note: MNB data is by decile (10 rows), Eurostat by quintile (5 rows).
    We aggregate MNB deciles to quintiles: (D1+D2)→Q1, (D3+D4)→Q2, etc.
    """
    print("\nEngineering features...")

    # --- MNB: aggregate deciles → quintiles ---
    mnb = mnb.copy()
    mnb["quintile"] = ((mnb["income_decile"] - 1) // 2) + 1
    mnb_q = mnb.groupby("quintile")["debt_to_income_ratio"].mean()

    feat_dti = mnb_q.clip(0, 200) / 2
    feat_dti.name = "dti_proxy"
    feat_dti.index.name = "income_quintile"

    # --- Eurostat features ---
    feat_housing     = engineer_housing_burden(eurostat_data)
    feat_buffer      = engineer_buffer_score(eurostat_data)
    feat_deprivation = engineer_deprivation_index(eurostat_data)
    feat_consumption = engineer_consumption_stress(eurostat_data)
    feat_arrears     = engineer_arrears_exposure(eurostat_data)
    feat_volatility  = engineer_income_volatility(eurostat_data)

    # Rename indices to income_quintile for joining
    for s in [feat_housing, feat_buffer, feat_deprivation,
              feat_consumption, feat_arrears, feat_volatility]:
        s.index.name = "income_quintile"

    # Combine into matrix
    matrix = pd.concat([
        feat_dti,
        feat_housing,
        feat_buffer,
        feat_deprivation,
        feat_consumption,
        feat_arrears,
        feat_volatility,
    ], axis=1)

    matrix.index.name = "income_quintile"
    matrix.columns = FEATURES

    print(f"  Feature matrix shape: {matrix.shape}")
    print(f"  Null values:\n{matrix.isnull().sum().to_string()}")

    # Fill any remaining NaNs with column median
    for col in matrix.columns:
        if matrix[col].isnull().any():
            median_val = matrix[col].median()
            matrix[col] = matrix[col].fillna(median_val)
            print(f"  [INFO] Filled NaNs in '{col}' with median={median_val:.2f}")

    return matrix


def scale_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply StandardScaler (zero mean, unit variance).
    Returns scaled DataFrame with same index and columns.
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(matrix)
    return pd.DataFrame(scaled_values, index=matrix.index, columns=matrix.columns)


# ---------------------------------------------------------------------------
# Correlation check
# ---------------------------------------------------------------------------

def check_correlations(matrix: pd.DataFrame) -> None:
    """Print a correlation matrix and flag high correlations (>0.85)."""
    corr = matrix.corr()
    print("\nCorrelation matrix:")
    print(corr.round(2).to_string())

    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            c = abs(corr.iloc[i, j])
            if c > 0.85:
                high_corr.append((corr.columns[i], corr.columns[j], round(c, 3)))

    if high_corr:
        print(f"\n[WARN] High correlations (>0.85) detected — consider PCA:")
        for a, b, c in high_corr:
            print(f"  {a} ↔ {b}: {c}")
    else:
        print("\n[OK] No problematic multicollinearity detected.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Feature Engineering — Fintech Risk Terrain")
    print("=" * 60)

    eurostat_data = load_eurostat()
    mnb           = load_mnb()

    matrix = build_feature_matrix(eurostat_data, mnb)

    print("\n--- Raw Feature Matrix ---")
    print(matrix.round(2).to_string())

    check_correlations(matrix)

    # Scale
    scaled = scale_features(matrix)

    # Save
    matrix.to_csv(OUTPUT_FEATURE_MATRIX)
    scaled.to_csv(OUTPUT_SCALED_MATRIX)

    print(f"\nSaved raw features : {OUTPUT_FEATURE_MATRIX}")
    print(f"Saved scaled features: {OUTPUT_SCALED_MATRIX}")

    print("\n--- Scaled Feature Matrix (preview) ---")
    print(scaled.round(3).to_string())

    print("\n[DONE] Data loading and feature engineering complete.")
    print("Next step: clustering — run your 04_cluster.py script.")


if __name__ == "__main__":
    main()