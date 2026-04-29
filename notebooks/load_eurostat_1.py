"""
=============================================================================
DIGITAL INEQUALITY IN CENTRAL EUROPE
Step 1: Data Collection — Eurostat API
=============================================================================
Run this FIRST. It pulls 8 indicators from Eurostat one by one,
filters and cleans each before merging (no memory explosions).

Install deps:
    pip install eurostat pandas requests openpyxl

Run:
    python 01_data_collection.py
=============================================================================
"""

import time
import warnings
from pathlib import Path

import pandas as pd
import eurostat

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("../data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_YEAR = 2021   # most complete NUTS2 coverage year
YEAR_FALLBACK = [2021, 2020, 2019, 2022]  # try in this order

# Countries to keep (Central + Eastern Europe + bordering anchors)
# Set to None to keep all EU27
COUNTRIES = ["HU", "PL", "CZ", "SK", "AT", "RO", "BG", "SI", "HR", "DE"]


# =============================================================================
# DATASET DEFINITIONS
# Each entry:
#   code       → Eurostat dataset code
#   col        → name we give the extracted value column
#   filter_pars→ passed directly to eurostat.get_data_df to limit download size
#   desc       → human label for logging
# =============================================================================
DATASETS = [
    {
        "code": "nama_10r_2gdp",
        "col": "gdp_mio_eur",
        "desc": "GDP (million EUR)",
        "filter_pars": {"unit": "MIO_EUR"},
    },
    {
        "code": "nama_10r_2gdp",
        "col": "gdp_pps_per_inh",
        "desc": "GDP per inhabitant (PPS, EU=100)",
        "filter_pars": {"unit": "PPS_HAB_EU27_2020"},
    },
    {
        "code": "lfst_r_lfe2emprt",
        "col": "employment_rate",
        "desc": "Employment rate 20-64 (%)",
        "filter_pars": {"sex": "T", "age": "Y20-64"},
    },
    {
        "code": "edat_lfse_04",
        "col": "tertiary_edu_pct",
        "desc": "Tertiary education 25-64 (%)",
        "filter_pars": {"sex": "T", "age": "Y25-64", "isced11": "ED5-8"},
    },
    {
        "code": "rd_e_gerdreg",
        "col": "rd_pct_gdp",
        "desc": "R&D expenditure (% of GDP)",
        "filter_pars": {"sectperf": "TOTAL", "unit": "PC_GDP"},
    },
    {
        "code": "rd_p_persocc",
        "col": "rd_personnel_pct",
        "desc": "R&D personnel (% active pop)",
        "filter_pars": {"sex": "T", "prof_pos": "TOTAL",
                        "sectperf": "TOTAL", "unit": "PC_ACT_POP"},
    },
    {
        "code": "isoc_r_broad_h",
        "col": "broadband_pct",
        "desc": "Broadband access - households (%)",
        "filter_pars": {"unit": "PC_HH"},
    },
    {
        "code": "isoc_r_iuse_i",
        "col": "internet_use_pct",
        "desc": "Internet use - individuals (%)",
        "filter_pars": {"unit": "PC_IND", "indic_is": "I_IU3",
                        "ind_type": "IND_TOTAL"},
    },
    {
        "code": "demo_r_pjanaggr3",
        "col": "population",
        "desc": "Population (persons)",
        "filter_pars": {"sex": "T", "age": "TOTAL"},
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def pick_year_column(df, years):
    """Return the first matching year column found in df."""
    for y in years:
        matches = [c for c in df.columns if str(y) in str(c)]
        if matches:
            return matches[0], y
    return None, None


def extract_nuts2_value(df_raw, col_name):
    """
    Given a raw wide DataFrame from eurostat.get_data_df:
      - find the geo column
      - filter to NUTS2 codes (exactly 4 chars)
      - pick the best available year
      - return a clean 2-column DataFrame [nuts2, col_name]
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["nuts2", col_name])

    # ── 1. Find the geo/region column ─────────────────────────────────────────
    # Eurostat puts geo+time in one column like "geo\time" or just "geo"
    geo_col = None
    for c in df_raw.columns:
        if "geo" in str(c).lower() or "nuts" in str(c).lower():
            geo_col = c
            break

    if geo_col is None:
        # Last resort: find the first column where values look like region codes
        for c in df_raw.columns:
            sample = df_raw[c].dropna().head(5)
            if sample.dtype == object and any(len(str(v)) in (2, 3, 4, 5) for v in sample):
                geo_col = c
                break

    if geo_col is None:
        print(f"    Could not identify geo column. Cols: {df_raw.columns.tolist()[:8]}")
        return pd.DataFrame(columns=["nuts2", col_name])

    df = df_raw.copy()
    df = df.rename(columns={geo_col: "nuts2"})
    df["nuts2"] = df["nuts2"].astype(str).str.strip()

    # ── 2. Filter to NUTS2 (exactly 4 characters: 2 country + 2 region) ───────
    df = df[df["nuts2"].str.len() == 4].copy()
    if df.empty:
        print(f"    No NUTS2 rows after length filter")
        return pd.DataFrame(columns=["nuts2", col_name])

    # ── 3. Pick year column ────────────────────────────────────────────────────
    year_col, used_year = pick_year_column(df, YEAR_FALLBACK)
    if year_col is None:
        year_cols_found = [c for c in df.columns if str(c).isdigit()]
        print(f"    No year column found. Year cols present: {year_cols_found[:5]}")
        return pd.DataFrame(columns=["nuts2", col_name])

    if used_year != TARGET_YEAR:
        print(f"    Year {TARGET_YEAR} missing -> using {used_year}")

    # ── 4. Extract and clean ───────────────────────────────────────────────────
    out = df[["nuts2", year_col]].copy()
    out.columns = ["nuts2", col_name]
    out[col_name] = pd.to_numeric(out[col_name], errors="coerce")
    out = out.dropna(subset=[col_name])
    out = out.drop_duplicates(subset="nuts2")

    return out.reset_index(drop=True)


# =============================================================================
# MAIN FETCH LOOP
# =============================================================================

def fetch_all():
    master = None
    print("\n" + "="*60)
    print("  EUROSTAT DATA PULL")
    print("="*60)

    for i, ds in enumerate(DATASETS, 1):
        code = ds["code"]
        col  = ds["col"]
        desc = ds["desc"]
        fp   = ds["filter_pars"]

        print(f"\n[{i}/{len(DATASETS)}] {col}")
        print(f"    {desc}")
        print(f"    Dataset: {code} | Filters: {fp}")

        try:
            raw = eurostat.get_data_df(
                code,
                flags=False,
                filter_pars=fp
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            print(f"    Check dataset code at: https://ec.europa.eu/eurostat/data/database")
            continue

        if raw is None:
            print(f"    Empty response from API")
            continue

        print(f"    Raw shape: {raw.shape}")

        # Extract just the NUTS2 + value we need — no big DataFrames held in memory
        chunk = extract_nuts2_value(raw, col)

        # Free raw memory immediately
        del raw

        if chunk.empty:
            print(f"    No usable NUTS2 data extracted")
            continue

        # Filter to target countries
        if COUNTRIES:
            chunk = chunk[chunk["nuts2"].str[:2].isin(COUNTRIES)].copy()

        print(f"    OK: {len(chunk)} regions")

        # Merge into master — chunk is tiny (max ~240 rows x 2 cols)
        if master is None:
            master = chunk
        else:
            master = master.merge(chunk, on="nuts2", how="outer")

        # Be polite to the API
        time.sleep(1.0)

    return master


# =============================================================================
# REPORT
# =============================================================================

def report(df):
    print("\n" + "="*60)
    print("  FINAL DATASET SUMMARY")
    print("="*60)
    print(f"  Regions : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")

    print(f"\n  {'Column':<30} {'Non-null':>8}  {'Missing %':>9}")
    print("  " + "-"*52)
    for col in df.columns:
        nn   = df[col].notna().sum()
        miss = 100.0 * (1 - nn / len(df))
        warn = " <-- HIGH MISSING" if miss > 40 else ""
        print(f"  {col:<30} {nn:>8}  {miss:>8.1f}%{warn}")

    # Check Budapest is present
    bud = df[df["nuts2"] == "HU11"]
    print(f"\n  Budapest (HU11): {'FOUND' if len(bud) > 0 else 'MISSING'}")
    if len(bud) > 0:
        print(bud.to_string(index=False))


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    master = fetch_all()

    if master is None or master.empty:
        print("\nNo data collected. Check your internet connection.")
        print("All calls go to: https://ec.europa.eu/eurostat/api/")
        raise SystemExit(1)

    # Add country column and Budapest flag
    master["country"] = master["nuts2"].str[:2]
    master["is_budapest"] = (master["nuts2"] == "HU11").astype(int)

    # Save
    out_path = OUTPUT_DIR / "eurostat_nuts2.csv"
    master.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    report(master)

    print("""
--------------------------------------------------------------
  NEXT STEPS
--------------------------------------------------------------
  1. Download RCI 2022 data (required for next step):
     https://cohesiondata.ec.europa.eu/Other/
       EU-RCI-2-0-2022-Time-Evolution/ngxn-bw8m/about_data
     -> Export -> CSV -> save as: data/raw/rci_2022.csv

  2. Download NUTS2 shapefile (needed for maps later):
     https://gisco-services.ec.europa.eu/distribution/v2/
       nuts/geojson/NUTS_RG_20M_2021_4326.geojson
     -> save as: data/raw/nuts2.geojson

  Then paste the output above here and we will move
  to step 2: merging RCI + preprocessing.
--------------------------------------------------------------
""")