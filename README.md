Fintech Risk Terrain — Project Structure
fintech_risk_terrain/
├── README.md
├── requirements.txt
├── config.py               # All dataset codes, paths, constants
├── 01_load_eurostat.py     # Pull all Eurostat tables via API
├── 02_load_mnb.py          # Parse MNB PDF tables (household debt data)
├── 03_merge_features.py    # Join all sources, engineer features
└── data/
├── raw/                # Auto-created — raw API responses
└── processed/          # Auto-created — cleaned, merged output
