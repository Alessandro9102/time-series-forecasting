"""
=============================================================================
DIGITAL INEQUALITY IN CENTRAL EUROPE
Step 2: Load RCI 2022 + NUTS2 Shapefile, merge with Eurostat data
=============================================================================

BEFORE RUNNING THIS SCRIPT you need two manual downloads:

  A) RCI 2022 CSV
     1. Go to: https://cohesiondata.ec.europa.eu/Other/
                EU-RCI-2-0-2022-Time-Evolution/ngxn-bw8m/about_data
     2. Click the "Export" button (top right) -> "CSV"
     3. Save as: data/raw/rci_2022.csv

  B) NUTS2 GeoJSON (for maps in later steps)
     1. Open this URL directly in your browser:
        https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326.geojson
     2. It will download automatically (it is ~10 MB)
     3. Save as: data/raw/nuts2.geojson

Install deps (if not already done):
    pip install pandas geopandas

Run:
    python 02_load_rci.py
=============================================================================
"""

import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW    = Path("data/raw")
DATA_PROC   = Path("data/processed")
DATA_PROC.mkdir(parents=True, exist_ok=True)

EUROSTAT_FILE = DATA_RAW / "eurostat_nuts2.csv"
RCI_FILE      = DATA_RAW / "rci_2022.csv"
GEOJSON_FILE  = DATA_RAW / "nuts2.geojson"

# Countries in scope
COUNTRIES = ["HU", "PL", "CZ", "SK", "AT", "RO", "BG", "SI", "HR", "DE"]

# RCI has 3 sub-indices + 11 pillars. These are the columns we want to keep.
# The script auto-detects names, but this is what we expect:
#   overall RCI score, Basic sub-index, Efficiency sub-index, Innovation sub-index
#   + the 11 pillars (optional, for deeper analysis)
RCI_SUBINDICES = ["basic", "efficiency", "innovation"]
RCI_PILLARS = [
    "institutions", "macroeconomic_stability", "infrastructure",
    "health", "basic_education", "higher_education",
    "labour_market", "market_size", "technological_readiness",
    "business_sophistication", "innovation_pillar"
]


# =============================================================================
# STEP 1 — INSPECT THE RCI CSV
# The Cohesion Open Data export can have different column names depending on
# the export format. This function figures out the structure automatically.
# =============================================================================

def inspect_rci(path: Path) -> pd.DataFrame:
    """Load, inspect and normalise the RCI CSV."""
    print(f"\n[1/3] LOADING RCI FILE")
    print(f"      {path}")
    print("─" * 60)

    if not path.exists():
        print("""
  FILE NOT FOUND!
  
  Download it here:
    https://cohesiondata.ec.europa.eu/Other/
      EU-RCI-2-0-2022-Time-Evolution/ngxn-bw8m/about_data
  -> Click Export -> CSV
  -> Save as: data/raw/rci_2022.csv
""")
        return pd.DataFrame()

    # Try reading with different encodings (EU open data sometimes uses latin-1)
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    print(f"  Raw shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"    {c}")

    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())

    return df


def clean_rci(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names and extract the NUTS2 code + key RCI scores.
    Returns a clean DataFrame with columns: nuts2, rci_score, rci_basic,
    rci_efficiency, rci_innovation (+ any pillars found).
    """
    if df_raw.empty:
        return pd.DataFrame()

    print(f"\n[2/3] CLEANING RCI DATA")
    print("─" * 60)

    df = df_raw.copy()

    # ── Normalise column names ─────────────────────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    print(f"  Normalised column names: {df.columns.tolist()}")

    # ── Find the NUTS2 code column ─────────────────────────────────────────────
    nuts_candidates = [c for c in df.columns
                       if any(kw in c for kw in ["nuts", "geo", "region", "code", "id"])]
    print(f"\n  NUTS column candidates: {nuts_candidates}")

    nuts_col = None
    for c in nuts_candidates:
        # Check if values look like NUTS2 codes (2 letters + 2 chars)
        sample = df[c].dropna().astype(str).head(20)
        if any(len(v.strip()) == 4 and v[:2].isalpha() for v in sample):
            nuts_col = c
            print(f"  -> Using '{c}' as NUTS2 column")
            break

    if nuts_col is None:
        # Last resort: scan all object columns
        for c in df.select_dtypes(include="object").columns:
            sample = df[c].dropna().astype(str).head(20)
            if any(len(v.strip()) == 4 and v[:2].isalpha() for v in sample):
                nuts_col = c
                print(f"  -> Using '{c}' as NUTS2 column (fallback)")
                break

    if nuts_col is None:
        print("  ERROR: Cannot find a NUTS2 code column.")
        print("  Please check the file and set nuts_col manually.")
        return pd.DataFrame()

    df = df.rename(columns={nuts_col: "nuts2"})
    df["nuts2"] = df["nuts2"].astype(str).str.strip().str.upper()

    # Keep only proper NUTS2 codes (4 characters)
    before = len(df)
    df = df[df["nuts2"].str.len() == 4].copy()
    print(f"  Filtered to NUTS2 (4-char codes): {before} -> {len(df)} rows")

    # ── Find RCI score columns ─────────────────────────────────────────────────
    # The overall RCI score column often contains "overall", "rci", "score", "index"
    score_candidates = [c for c in df.columns
                        if any(kw in c for kw in ["overall", "score", "rci", "index",
                                                   "basic", "effic", "innov"])]
    print(f"\n  Score column candidates: {score_candidates}")

    # Try to find the year 2022 columns if multiple years exist
    year_cols = [c for c in df.columns if "2022" in c]
    if year_cols:
        print(f"  Year 2022 columns: {year_cols}")

    # ── Build clean output ─────────────────────────────────────────────────────
    # We'll keep nuts2 + all numeric columns (let user inspect)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    keep_cols = ["nuts2"] + numeric_cols
    out = df[keep_cols].copy()

    # Try to rename the main score columns to standard names
    # Strategy: look for columns whose name contains key terms
    rename_map = {}
    for c in out.columns:
        cl = c.lower()
        if "overall" in cl or (("rci" in cl or "score" in cl) and "sub" not in cl
                                and "basic" not in cl and "effic" not in cl and "innov" not in cl):
            rename_map[c] = "rci_score"
        elif "basic" in cl and "edu" not in cl:
            rename_map[c] = "rci_basic"
        elif "effic" in cl:
            rename_map[c] = "rci_efficiency"
        elif "innov" in cl:
            rename_map[c] = "rci_innovation"

    if rename_map:
        out = out.rename(columns=rename_map)
        print(f"\n  Renamed columns: {rename_map}")
    else:
        print("\n  Could not auto-rename score columns.")
        print("  All numeric columns kept as-is — inspect and rename manually if needed.")

    # Remove duplicate NUTS2 rows (keep the 2022 entry if multiple years)
    out = out.drop_duplicates(subset="nuts2")

    print(f"\n  Clean RCI shape: {out.shape}")
    print(f"  Columns: {out.columns.tolist()}")

    # Show Budapest
    bud = out[out["nuts2"] == "HU11"]
    print(f"\n  Budapest (HU11):")
    if len(bud) > 0:
        print(bud.to_string(index=False))
    else:
        print("  NOT FOUND — check NUTS code format in the CSV")

    return out


# =============================================================================
# STEP 2 — VERIFY THE GEOJSON (just a sanity check, used in Step 4 for maps)
# =============================================================================

def check_geojson(path: Path):
    print(f"\n[Optional] CHECKING NUTS2 GEOJSON")
    print("─" * 60)

    if not path.exists():
        print(f"""
  FILE NOT FOUND: {path}

  Download it by opening this URL in your browser:
    https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326.geojson
  It downloads automatically (~10 MB).
  Save as: data/raw/nuts2.geojson
""")
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        features = gj.get("features", [])
        # Count NUTS2 features
        nuts2_features = [
            feat for feat in features
            if feat.get("properties", {}).get("LEVL_CODE") == 2
        ]
        print(f"  Total features: {len(features)}")
        print(f"  NUTS2 features: {len(nuts2_features)}")

        # Check a HU region
        hu_regions = [
            feat["properties"]["NUTS_ID"]
            for feat in nuts2_features
            if feat["properties"]["NUTS_ID"].startswith("HU")
        ]
        print(f"  Hungarian NUTS2 regions: {hu_regions}")
        print(f"  GeoJSON OK")
        return True
    except Exception as e:
        print(f"  Error reading GeoJSON: {e}")
        return False


# =============================================================================
# STEP 3 — MERGE EUROSTAT + RCI INTO ONE MASTER FILE
# =============================================================================

def merge_datasets(eurostat_df: pd.DataFrame, rci_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[3/3] MERGING EUROSTAT + RCI")
    print("─" * 60)

    if eurostat_df.empty:
        print("  ERROR: Eurostat data is empty. Run 01_data_collection.py first.")
        return pd.DataFrame()

    print(f"  Eurostat shape: {eurostat_df.shape}")
    print(f"  RCI shape:      {rci_df.shape if not rci_df.empty else 'EMPTY (will skip)'}")

    master = eurostat_df.copy()

    if not rci_df.empty:
        # Merge on NUTS2 code — left join keeps all Eurostat regions
        before = len(master)
        master = master.merge(rci_df, on="nuts2", how="left")
        rci_cols = [c for c in rci_df.columns if c != "nuts2"]
        matched = master[rci_cols[0]].notna().sum() if rci_cols else 0
        print(f"  Merged: {before} Eurostat regions, {matched} matched to RCI")

        # Warn about unmatched regions
        unmatched = master[master[rci_cols[0]].isna()]["nuts2"].tolist() if rci_cols else []
        if unmatched:
            print(f"  Unmatched regions (RCI missing): {unmatched[:10]}")
    else:
        print("  Skipping RCI merge (file not found)")

    # ── Final missingness report ───────────────────────────────────────────────
    print(f"\n  MASTER DATASET SUMMARY")
    print(f"  {'Column':<35} {'Non-null':>8}  {'Missing %':>9}")
    print(f"  {'─'*57}")
    for col in master.columns:
        nn   = master[col].notna().sum()
        miss = 100.0 * (1 - nn / len(master))
        warn = "  <-- HIGH" if miss > 40 else ""
        print(f"  {col:<35} {nn:>8}  {miss:>8.1f}%{warn}")

    # Save
    out_path = DATA_PROC / "master_merged.csv"
    master.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    print(f"  Final shape: {master.shape[0]} regions x {master.shape[1]} columns")

    return master


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 2: LOAD RCI + MERGE")
    print("=" * 60)

    # Load Eurostat data from step 1
    if not EUROSTAT_FILE.exists():
        print(f"\nERROR: {EUROSTAT_FILE} not found.")
        print("Run 01_data_collection.py first.")
        raise SystemExit(1)

    eurostat_df = pd.read_csv(EUROSTAT_FILE)
    print(f"\nLoaded Eurostat data: {eurostat_df.shape}")

    # Load + clean RCI
    rci_raw = inspect_rci(RCI_FILE)
    rci_clean = clean_rci(rci_raw)

    # Check GeoJSON
    check_geojson(GEOJSON_FILE)

    # Merge everything
    master = merge_datasets(eurostat_df, rci_clean)

    print("""
--------------------------------------------------------------
  WHAT TO DO NEXT
--------------------------------------------------------------
  Paste the output above here (especially the MASTER DATASET
  SUMMARY table) so we can see:
    - how many regions were matched between Eurostat and RCI
    - which columns have high missing values (>40%)
    - whether Budapest HU11 is present in both sources

  Once we confirm the merge looks good, we move to:
    Step 3: 03_preprocessing.py
      - handle missing values (KNN imputation)
      - standardise all features (StandardScaler)
      - build the final 230 x 15 feature matrix
--------------------------------------------------------------
""")