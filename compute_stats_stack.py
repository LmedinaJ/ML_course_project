import numpy as np
import pandas as pd
import os
from glob import glob

basePath = 'npz_files_stack'
files = glob(os.path.join(basePath, '*.npz'))

stats_list = []

for file_path in files:
    try:
        with np.load(file_path) as npz:
            keys = list(npz.keys())
            if 'data' in keys:
                data = npz['data']
            elif len(keys) == 1:
                data = npz[keys[0]]
            else:
                raise ValueError(f"Multiple arrays found in {file_path}, and no key='data'")

            # Expecting shape: (49, 192, 192, 2)
            if data.shape[-1] != 2:
                raise ValueError(f"Expected 2 channels, got shape {data.shape} in {file_path}")

            min_vals = data.min(axis=(0, 1, 2))  # shape (2,)
            max_vals = data.max(axis=(0, 1, 2))  # shape (2,)
            mean_vals = data.mean(axis=(0, 1, 2))
            std_vals = data.std(axis=(0, 1, 2))

            stats_list.append({
                'file_name': os.path.basename(file_path),
                'min_ch0': min_vals[0],
                'max_ch0': max_vals[0],
                'mean_ch0': mean_vals[0],
                'std_ch0': std_vals[0],
                'min_ch1': min_vals[1],
                'max_ch1': max_vals[1],
                'mean_ch1': mean_vals[1],
                'std_ch1': std_vals[1]
            })

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")

# Save results
df_stats = pd.DataFrame(stats_list)
os.makedirs('csv_data', exist_ok=True)
df_stats.to_csv('csv_data/per_channel_npz_stats.csv', index=False)

print("✅ Done! Stats saved to 'csv_data/per_channel_npz_stats.csv'")
