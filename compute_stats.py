from glob import glob
import os, numpy as np, pandas as pd

BASE_DIR     = "npz_files"
OUT_DIR      = "csv_data"
os.makedirs(OUT_DIR, exist_ok=True)
stats_path   = os.path.join(OUT_DIR, "npz_file_stats.csv")

rows = []

for fp in glob(os.path.join(BASE_DIR, "*.npz")):
    try:
        npz = np.load(fp)
    except Exception as e:
        print(f"⚠️  Could not open {fp}: {e}")
        continue

    # iterate over every array stored in the file
    for key in npz.files:                       # .files is guaranteed order‑safe
        data = npz[key]
        fname = os.path.basename(fp)

        # ---- file‑name tokens ------------------------------------------------
        parts  = fname.split("_")
        event_id = parts[0]                     # R1803… or S77… etc.
        sensor   = parts[1]                     # ir069 / ir107 / vis …
        fi       = parts[2]                     # fiXXX
        event    = parts[3]                     # Flood / FlashFlood / random …

        rows.append({
            "file_name": fname,
            "array_key": key,                   # keep track of which key
            "event_id":  event_id,
            "sensor":    sensor,
            "fi":        fi,
            "event":     event,
            "min":  data.min().item(),
            "max":  data.max().item(),
            "mean": data.mean().item(),
            "std":  data.std().item(),
        })

    npz.close()  # close the file handle

df_stats = pd.DataFrame(rows)
df_stats.to_csv(stats_path, index=False)
print(f"✅ Stats written to {stats_path}  |  rows = {len(df_stats)}")
