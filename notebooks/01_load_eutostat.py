"""
01_load_eurostat.py
-------------------
Downloads all Eurostat datasets defined in config.py using the
`eurostat` Python package (wraps the Eurostat JSON API).

Run:
    python 01_load_eurostat.py

Output:
    data/raw/eurostat_raw.pkl   — dict of {dataset_code: DataFrame}
    data/raw/{code}.csv         — individual CSVs for quick inspection
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import time
import warnings

import eurostat
import pandas as pd
from tqdm import tqdm

from config import (
    COMPARISON_COUNTRIES,
    EUROSTAT_DATASETS,
    OUTPUT_EUROSTAT_RAW,
    DATA_RAW,
    YEAR_START,
    YEAR_END,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_dataset(code: str, meta: dict) -> pd.DataFrame:
    """
    Pull a single Eurostat dataset by code, apply country filter,
    and return a tidy long-format DataFrame.

    Parameters
    ----------
    code : str
        Eurostat dataset code, e.g. 'ilc_mdes09'
    meta : dict
        Entry from EUROSTAT_DATASETS — contains label and filter info

    Returns
    -------
    pd.DataFrame with at minimum columns: geo, time, values
    """
    print(f"\n  Fetching: {code}  ({meta['label']})")

    try:
        # Pull full dataset — the eurostat package returns a wide DataFrame
        df_wide = eurostat.get_data_df(code)
    except Exception as e:
        print(f"  [ERROR] Could not fetch {code}: {e}")
        return pd.DataFrame()

    if df_wide is None or df_wide.empty:
        print(f"  [WARN] Empty response for {code}")
        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # The eurostat package returns columns like:
    #   freq | geo\TIME_PERIOD | 2015 | 2016 | ... | 2023
    # We need to melt to long format: geo | time | values
    # -----------------------------------------------------------------------
    df = df_wide.copy()

    # Identify the geo column (it often has a backslash in the name)
    geo_col = [c for c in df.columns if "geo" in c.lower() or "TIME" in c]
    if not geo_col:
        print(f"  [WARN] Could not identify geo column in {code}. Columns: {list(df.columns)}")
        return pd.DataFrame()

    geo_col = geo_col[0]

    # Identify year columns (numeric strings)
    year_cols = [
        c for c in df.columns
        if str(c).strip().isdigit()
        and YEAR_START <= int(str(c).strip()) <= YEAR_END
    ]

    if not year_cols:
        print(f"  [WARN] No year columns found for range {YEAR_START}–{YEAR_END} in {code}")
        return pd.DataFrame()

    # Melt to long format
    id_vars = [c for c in df.columns if c not in year_cols]
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols,
                      var_name="time", value_name="values")

    # Rename geo column for consistency
    df_long = df_long.rename(columns={geo_col: "geo"})

    # Clean up geo values (strip whitespace)
    df_long["geo"] = df_long["geo"].astype(str).str.strip()

    # Filter to target countries
    df_long = df_long[df_long["geo"].isin(COMPARISON_COUNTRIES)].copy()

    # Convert time to integer year
    df_long["time"] = pd.to_numeric(df_long["time"], errors="coerce").astype("Int64")

    # Convert values to numeric (Eurostat uses ':' for missing)
    df_long["values"] = pd.to_numeric(
        df_long["values"].astype(str).str.replace(":", "", regex=False).str.strip(),
        errors="coerce",
    )

    # Drop rows where both key dims are null
    df_long = df_long.dropna(subset=["geo", "time"])

    print(f"  OK — {len(df_long):,} rows, {df_long['geo'].nunique()} countries, "
          f"years {df_long['time'].min()}–{df_long['time'].max()}")

    return df_long


def summarise_coverage(all_data: dict) -> pd.DataFrame:
    """
    Print a quick coverage table: how many non-null HU values per dataset.
    """
    rows = []
    for code, df in all_data.items():
        if df.empty:
            rows.append({"dataset": code, "hu_rows": 0, "non_null": 0, "years": "—"})
            continue
        hu = df[df["geo"] == "HU"]
        non_null = hu["values"].notna().sum()
        years = (
            f"{int(hu['time'].min())}–{int(hu['time'].max())}"
            if not hu.empty else "—"
        )
        rows.append({"dataset": code, "hu_rows": len(hu),
                     "non_null": non_null, "years": years})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Eurostat Data Loader — Fintech Risk Terrain")
    print("=" * 60)
    print(f"Target countries : {COMPARISON_COUNTRIES}")
    print(f"Year range       : {YEAR_START}–{YEAR_END}")
    print(f"Datasets to fetch: {len(EUROSTAT_DATASETS)}")
    print()

    all_data = {}

    for code, meta in tqdm(EUROSTAT_DATASETS.items(), desc="Datasets", unit="ds"):
        df = fetch_dataset(code, meta)
        all_data[code] = df

        # Save individual CSV for easy inspection in VS Code
        if not df.empty:
            csv_path = DATA_RAW / f"{code}.csv"
            df.to_csv(csv_path, index=False)

        # Be polite to the Eurostat API — avoid rate limiting
        time.sleep(1.5)

    # Save full dict as pickle for fast loading in downstream scripts
    with open(OUTPUT_EUROSTAT_RAW, "wb") as f:
        pickle.dump(all_data, f)

    print(f"\n\nAll datasets saved to: {OUTPUT_EUROSTAT_RAW}")

    # Coverage report
    print("\n--- Coverage Report (Hungary) ---")
    coverage = summarise_coverage(all_data)
    print(coverage.to_string(index=False))

    # Flag any empty datasets
    failed = [code for code, df in all_data.items() if df.empty]
    if failed:
        print(f"\n[WARN] The following datasets returned no data: {failed}")
        print("  Check the dataset codes at: https://ec.europa.eu/eurostat/data/database")
    else:
        print("\n[OK] All datasets loaded successfully.")

    print("\nNext step: run python 02_load_mnb.py")


if __name__ == "__main__":
    main()