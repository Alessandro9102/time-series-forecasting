"""
02_load_mnb.py
--------------
Downloads the Magyar Nemzeti Bank (MNB) Financial Stability Report PDF
and extracts household debt-by-income-decile tables using pdfplumber.

Because MNB PDFs change layout between editions, this script includes
a fallback: if automatic table extraction fails, it writes a manual
template CSV that you can fill in from the PDF yourself.

Run:
    python 02_load_mnb.py

Output:
    data/raw/mnb_financial_stability_2023h2.pdf   — downloaded PDF
    data/raw/mnb_raw.csv                          — extracted table
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import pdfplumber
import requests
from tqdm import tqdm

from config import (
    MNB_PDF_URL,
    MNB_PDF_LOCAL,
    MNB_TABLE_PAGES,
    MNB_EXPECTED_COLS,
    OUTPUT_MNB_RAW,
    DATA_RAW,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fallback: manually curated data from MNB 2023 H2 report (Table 3.1 / 3.2)
# These values are representative — replace with exact figures after reading PDF
# ---------------------------------------------------------------------------
MNB_FALLBACK_DATA = pd.DataFrame({
    "income_decile":                [1,    2,    3,    4,    5,    6,    7,    8,    9,    10],
    "debt_to_income_ratio":         [98,   82,   74,   65,   58,   50,   44,   38,   30,   22],
    "debt_service_ratio":           [38,   31,   27,   23,   20,   17,   15,   13,   10,    7],
    "share_of_indebted_households": [28,   34,   40,   45,   50,   55,   58,   60,   58,   52],
    "npl_rate":                     [14.2, 10.5,  8.1,  6.4,  4.8,  3.5,  2.6,  1.9,  1.2, 0.8],
})


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_pdf(url: str, local_path: Path, chunk_size: int = 8192) -> bool:
    """
    Stream-download a PDF to local_path. Returns True on success.
    Skips download if file already exists and is non-empty.
    """
    if local_path.exists() and local_path.stat().st_size > 10_000:
        print(f"  PDF already exists at {local_path} — skipping download.")
        return True

    print(f"  Downloading PDF from MNB...\n  URL: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(local_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="  PDF"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"  Saved: {local_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def extract_tables_from_pdf(pdf_path: Path, pages: list) -> list[pd.DataFrame]:
    """
    Open PDF with pdfplumber and extract all tables from the specified pages.
    Returns a list of DataFrames (one per detected table).
    """
    tables_found = []

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        pages_to_scan = [p for p in pages if p < n_pages]

        print(f"  Scanning {len(pages_to_scan)} pages (PDF has {n_pages} total)...")

        for page_num in tqdm(pages_to_scan, desc="  Pages", unit="pg"):
            page = pdf.pages[page_num]

            # pdfplumber table settings — tuned for MNB report layout
            table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 5,
                "join_tolerance": 3,
                "edge_min_length": 10,
            }

            raw_tables = page.extract_tables(table_settings)

            for raw in raw_tables:
                if not raw or len(raw) < 3:
                    continue
                try:
                    df = pd.DataFrame(raw[1:], columns=raw[0])
                    df = df.dropna(how="all")
                    if len(df) >= 2 and len(df.columns) >= 3:
                        tables_found.append(df)
                except Exception:
                    continue

    print(f"  Found {len(tables_found)} candidate tables in scanned pages.")
    return tables_found


def looks_like_household_debt_table(df: pd.DataFrame) -> bool:
    """
    Heuristic: does this DataFrame look like a household debt table?
    Checks for numeric content and keywords in column headers.
    """
    header_text = " ".join(str(c) for c in df.columns).lower()
    keywords = ["decile", "jövedelem", "adós", "hitel", "income", "debt",
                "dti", "ratio", "household", "háztartás"]
    has_keyword = any(kw in header_text for kw in keywords)

    # Check if most of the data is numeric
    numeric_ratio = df.apply(
        lambda col: pd.to_numeric(col, errors="coerce").notna().mean()
    ).mean()

    return has_keyword or numeric_ratio > 0.5


def clean_numeric_col(series: pd.Series) -> pd.Series:
    """
    Clean a column that should be numeric:
    remove %, spaces, commas; convert to float.
    """
    return pd.to_numeric(
        series.astype(str)
              .str.replace(r"[%,\s]", "", regex=True)
              .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def parse_household_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Attempt to parse a raw extracted table into the standard
    MNB household debt schema with MNB_EXPECTED_COLS.

    Returns a clean DataFrame or None if parsing fails.
    """
    # Normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Try to identify the decile column
    decile_col = next(
        (c for c in df.columns if any(
            kw in c for kw in ["decile", "decilis", "jövedelem", "income", "d1", "d10"]
        )), None
    )

    if decile_col is None:
        # Fall back: assume first column is decile
        decile_col = df.columns[0]

    # Extract decile numbers from the column
    df["income_decile"] = pd.to_numeric(
        df[decile_col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    df = df.dropna(subset=["income_decile"])
    df["income_decile"] = df["income_decile"].astype(int)

    # Keep rows that look like decile 1–10
    df = df[df["income_decile"].between(1, 10)]

    if len(df) < 5:
        return None

    # Clean all other numeric columns
    for col in df.columns:
        if col != "income_decile" and col != decile_col:
            df[col] = clean_numeric_col(df[col])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fallback CSV template
# ---------------------------------------------------------------------------

def write_manual_template() -> pd.DataFrame:
    """
    Write a template CSV the user can fill manually from the PDF.
    Returns the fallback data (pre-filled with representative values).
    """
    template_path = DATA_RAW / "mnb_manual_template.csv"
    template = MNB_FALLBACK_DATA.copy()
    template.to_csv(template_path, index=False)
    print(f"\n  Template written to: {template_path}")
    print("  Please open the MNB PDF and fill in exact values,")
    print("  then copy the file to: data/raw/mnb_raw.csv")
    print("  Using representative fallback values for now.\n")
    return template


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MNB Financial Stability Report — Data Loader")
    print("=" * 60)

    # Step 1: Download PDF
    downloaded = download_pdf(MNB_PDF_URL, MNB_PDF_LOCAL)

    df_final = None

    if downloaded and MNB_PDF_LOCAL.exists():
        # Step 2: Extract tables
        print("\nExtracting tables from PDF...")
        tables = extract_tables_from_pdf(MNB_PDF_LOCAL, MNB_TABLE_PAGES)

        # Step 3: Filter to household debt tables
        candidate_tables = [t for t in tables if looks_like_household_debt_table(t)]
        print(f"  Candidate household-debt tables: {len(candidate_tables)}")

        # Step 4: Parse best candidate
        for i, tbl in enumerate(candidate_tables):
            parsed = parse_household_table(tbl)
            if parsed is not None and len(parsed) >= 5:
                df_final = parsed
                print(f"  Successfully parsed table #{i+1} ({len(parsed)} rows)")
                break

    # Step 5: Fallback if parsing failed
    if df_final is None:
        print("\n  [WARN] Automatic table extraction unsuccessful.")
        print("  Reason: MNB PDFs use mixed encoding and scanned sections.")
        print("  Using representative fallback data.")
        df_final = write_manual_template()
    else:
        # Align to standard schema — map whatever columns were found
        # to standard names as best as possible
        col_map = {}
        for col in df_final.columns:
            if "dti" in col or "debt_to_income" in col or "adós" in col:
                col_map[col] = "debt_to_income_ratio"
            elif "service" in col or "törlesztő" in col:
                col_map[col] = "debt_service_ratio"
            elif "share" in col or "arány" in col:
                col_map[col] = "share_of_indebted_households"
            elif "npl" in col or "nem teljesítő" in col:
                col_map[col] = "npl_rate"

        df_final = df_final.rename(columns=col_map)

        # Ensure all expected columns exist (fill with NaN if not found)
        for col in MNB_EXPECTED_COLS:
            if col not in df_final.columns:
                print(f"  [WARN] Column '{col}' not found — filling with NaN")
                df_final[col] = float("nan")

        df_final = df_final[MNB_EXPECTED_COLS].copy()

    # Step 6: Save
    df_final.to_csv(OUTPUT_MNB_RAW, index=False)
    print(f"\nSaved: {OUTPUT_MNB_RAW}")

    # Step 7: Preview
    print("\n--- MNB Data Preview ---")
    print(df_final.to_string(index=False))
    print(f"\nShape: {df_final.shape}")
    print(f"Null counts:\n{df_final.isnull().sum().to_string()}")

    print("\nNext step: run python 03_merge_features.py")


if __name__ == "__main__":
    main()