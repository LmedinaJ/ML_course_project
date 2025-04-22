from glob import glob
import pandas as pd
import numpy as np
import os

basePath = 'npz_files'
files = glob(basePath+'/*.npz')

data_output = 'csv_data'
stats_path = os.path.join(data_output, 'npz_file_stats.csv')

stats_list = []

for file_path in files:
    try:
        with np.load(file_path) as npz:
            keys = list(npz.keys())
            if len(keys) != 1:
                raise ValueError(f"Expected 1 array, got {len(keys)} in {file_path}")
            data = npz[keys[0]]

            stats_list.append({
                'file_name': file_path.split('/')[-1],
                'event_id': file_path.split('/')[-1].split('_')[0],
                'sensor': file_path.split('/')[-1].split('_')[1],
                'fi': file_path.split('/')[-1].split('_')[2],
                'event': file_path.split('/')[-1].split('_')[3],
                'min': np.min(data),
                'max': np.max(data),
                'mean': np.mean(data),
                'std': np.std(data)
            })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

df_stats = pd.DataFrame(stats_list)
df_stats.to_csv(stats_path,index=False)