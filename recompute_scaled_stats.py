# file: recompute_scaled_stats.py
from glob import glob
import os, sys
import numpy as np
import pandas as pd

RAW_DIR = "npz_files"                      # folder with per‑sensor .npz
OUT_CSV = "csv_data/npz_scaled_stats.csv"
os.makedirs("csv_data", exist_ok=True)

SCALE = {"vis": 1.0e-4, "ir069": 1.0e-2, "ir107": 1.0e-2}

rows = []
for fp in glob(os.path.join(RAW_DIR, "*.npz")):
    fname  = os.path.basename(fp)                     # S780109_ir107_fi389_…
    tokens = fname.replace(".npz", "").split("_")

    if len(tokens) < 4:                               # safety check
        print(f"⚠️  Unexpected name format: {fname}", file=sys.stderr)
        continue

    event_id   = tokens[0]        # S780109
    sensor     = tokens[1]        # ir107
    event      = tokens[3]        # FlashFlood / Flood / HeavyRain / random

    if sensor not in SCALE:
        continue                  # skip non‑raster .npz (e.g. metadata)

    arr  = np.load(fp)[sensor].astype(np.float32) * SCALE[sensor]
    rows.append({
        "event_id": event_id,
        "event":    event,
        "sensor":   sensor,
        "file_name": fname,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    })

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"✅  wrote {OUT_CSV}")
