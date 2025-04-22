from datetime import datetime
from zoneinfo import ZoneInfo
from glob import glob
import pandas as pd
import subprocess
import numpy as np
import random
import time
import os
import h5py # needs conda/pip install h5py

data_output = 'csv_data'
catalog_path = os.path.join(data_output, 'CATALOG.csv')
os.makedirs(data_output, exist_ok=True)

get_catalog = [
    'aws', 's3', 'cp',
    '--no-sign-request',
    's3://sevir/CATALOG.csv',
    catalog_path
]

if not os.path.exists(catalog_path):
    subprocess.run(get_catalog, check=True)

print('done with download of data catalog')

df = pd.read_csv('csv_data/CATALOG.csv')
random.seed(42)

# # Parameters
event_types = ['Flash Flood', 'Heavy Rain', 'Flood']
img_types = ['vis', 'ir107', 'ir069']
n_samples = 1000
nan_fraction = 0.5

# # Load and initial filter
# df = pd.read_csv('CATALOG.csv', low_memory=False)

filtered_df = df[
    (df['event_type'].isin(event_types) | df['event_type'].isna()) &
    (df['img_type'].isin(img_types))
]

# # Get unique event IDs with their event types
event_type_by_id = filtered_df.groupby('id')['event_type'].agg(lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else np.nan)

# # Separate IDs by event type explicitly
id_groups = {
    etype: event_type_by_id[event_type_by_id == etype].index.tolist()
    for etype in event_types
}
id_groups['nan'] = event_type_by_id[event_type_by_id.isna()].index.tolist()

# # Calculate sample counts per event type
n_nan = int(n_samples * nan_fraction)
n_known = n_samples - n_nan

# # Determine proportions for known event types from original data
known_counts = filtered_df['event_type'].value_counts(normalize=True)
sample_counts = (known_counts * n_known).round().astype(int).to_dict()

# # Adjust to exactly match n_known
difference = n_known - sum(sample_counts.values())
if difference != 0:
    sample_counts[event_types[0]] += difference  # Adjust to ensure sum exactly matches

# # Sample IDs
sampled_ids = []
for etype in event_types:
    sampled_ids += random.sample(id_groups[etype], min(sample_counts.get(etype, 0), len(id_groups[etype])))

sampled_ids += random.sample(id_groups['nan'], min(n_nan, len(id_groups['nan'])))

# # Final sampled DataFrame
random_sampled_df = filtered_df[filtered_df['id'].isin(sampled_ids)]

# # Display final proportions
print('Total number of unique event IDs:', event_type_by_id.size, '\n')
print("Original proportions:\n", filtered_df['event_type'].value_counts(normalize=True, dropna=False), '\n')

print('Sampled events:', len(sampled_ids), '\n')
print("Sampled proportions:\n", random_sampled_df['event_type'].value_counts(normalize=True, dropna=False))


# # sometimes some ids have less than the three required features
# # so we dont want to have any error in our training
# # so we are going to drop those rows
less_features = random_sampled_df.groupby('id').count()['file_name']==3
less_features[less_features]

ids_with_3_rows = random_sampled_df.groupby('id').count()['file_name'] == 3
ids_with_3_rows = ids_with_3_rows[ids_with_3_rows].index  # extract just the IDs

df_with_3_rows = random_sampled_df[random_sampled_df['id'].isin(ids_with_3_rows)]
order_download = df_with_3_rows['file_name'].value_counts().reset_index()#['file_name']

df_with_3_rows['time_utc'] = pd.to_datetime(df['time_utc'], format='%Y-%m-%d %H:%M:%S')
# df_with_3_rows.hist(  column='time_utc', bins=100, grid=False, figsize=(12,8), color='#86bf91', zorder=2)

# ## here i just want to check how many records i have from 8 am to 4 pm a
# ## and outsite that range
mask = (df_with_3_rows['time_utc'].dt.hour >= 8) & (df_with_3_rows['time_utc'].dt.hour < 16)

# ## separate the data
data_day = df_with_3_rows[mask]          # data between 8 AM and 4 PM
data_rest = df_with_3_rows[~mask]        # data outside the range

# # count the number samples
count_day = data_day.shape[0]
count_rest = data_rest.shape[0]

print(f"samples between 8 AM and 4 PM: {count_day}")
print(f"samples outside 8 AM to 4 PM: {count_rest}")

# # ir107/2018/SEVIR_IR107_STORMEVENTS_2018_0701_1231.h5
# order_download

def download_data(file):

  s3_url = f"s3://sevir/data/{file}"
  basePath = f"data_h5/{'/'.join(file.split('/')[0:2])}"
  # basePath = f"{drivePath}{'/'.join(file.split('/')[0:2])}"

  os.makedirs(basePath, exist_ok=True)
  local_path = os.path.join(basePath, os.path.basename(file))

  # Check if file already exists
  if os.path.exists(local_path):
      print(f"File {local_path} already exists. Skipping download.")
      return local_path

  print(f"Downloading: {s3_url} -> {local_path}")
  os.system(f"aws s3 cp --no-sign-request {s3_url} {local_path}")

  return local_path

def read_data( sample_event, img_type, fn, fi ):
    """
    Reads single SEVIR event for a given image type.

    Parameters
    ----------
    dataset   pd.DataFrame
        SEVIR catalog rows matching a single ID
    img_type   str
        SEVIR image type
    data_path  str
        Location of SEVIR data

    Returns
    -------
    np.array
       LxLx49 tensor containing event data
    """

    #fn = dataset[dataset.img_type==img_type].squeeze().file_name
    #fi = dataset[dataset.img_type==img_type].squeeze().file_index
    with h5py.File( fn,'r') as hf:
        data=hf[img_type][fi]#.astype(np.float64)

    return data


def print_time_now():
  now = datetime.now(ZoneInfo("Asia/Bangkok"))
  current_time = now.strftime("%H:%M:%S")
  return current_time

df_with_3_rows.to_csv('csv_data/data_to_download.csv')
order_download.to_csv('csv_data/order_download.csv')

data_2process = order_download['file_name']#][46:]
# len(data_2process)
# # for i in data_2process:
# #   print(i)

drivePath = 'npz_files/'
os.makedirs(drivePath, exist_ok=True)

print(f'time start: {print_time_now()}')
total_files = len(data_2process)
start_time = time.time()
# # for i in data_2process[0:1]:
for index, i in enumerate(data_2process, start=1):
  # files_left = total_files - index
  print(f'processing: {index}/{total_files} files')
  local_path = download_data(i)

  data_process = df_with_3_rows[df_with_3_rows['file_name']==i]

  for index, row in data_process.iterrows():
    weather_event = row['event_type'] if pd.notna(row['event_type']) else 'random'
    sensor        = row['img_type']
    file_index    = row['file_index']
    file_name     = row['file_name']
    event_id      = row['id']
    date          = row['time_utc']
    data_min      = row['data_min']
    data_max      = row['data_max']

    year          = file_name.split('_')[3]
    time_start    = file_name.split('_')[4]
    time_end      = file_name.split('_')[5].replace('.h5','')

    #SEVIR_IR069_RANDOMEVENTS_2019_0101_0430.h5

    name_export = f"{event_id}_{sensor}_fi{file_index}_{weather_event.replace(' ', '')}_{year}_{time_start}_{time_end}"
    array_sensor = read_data(local_path, sensor, local_path, file_index)

    basePath = f"{drivePath}{name_export}"

    # print('array_sensor \n',array_sensor)
    # print('array_sensor \n',name_export)
    if sensor == 'vis':
      np.savez(f'{basePath}.npz', vis=array_sensor)
    if sensor == 'ir069':
      np.savez(f'{basePath}.npz', ir069=array_sensor)
    if sensor == 'ir107':
      np.savez(f'{basePath}.npz', ir107=array_sensor)


    # print(event_id)
    # print(file_name)
    # print(file_index)
    # print(sensor)
    # print(weather_event)
    # print('--'*3,'\n')
  print('\n')
  os.remove(local_path)

print(f'--- processing time --- {(time.time() - start_time)/ 60}')
print(f'time finish: {print_time_now()}')
