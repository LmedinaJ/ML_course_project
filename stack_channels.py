from collections import defaultdict
import torch.nn.functional as F
from glob import glob
import pandas as pd
import numpy as np
import torch
import time
import os


basePath = 'npz_files'
files = glob(basePath+'/*.npz')

print('##'*5)
print(f'total exported files: {len(files)}')
print('##'*5)

# Create a defaultdict to group file paths by unique ID
grouped_files = defaultdict(list)

for filepath in files:
    # Extract the base name, split by underscore, and pick the first part as the unique ID.
    unique_id = os.path.basename(filepath).split('_')[0]
    grouped_files[unique_id].append(filepath)


df_stats = pd.read_csv("csv_data/npz_scaled_stats.csv")


sensor_minmax = (
    df_stats.groupby("sensor")[["min", "max"]]
            .agg({"min": "min", "max": "max"})
            .to_dict("index")
)

label_map = {
    'FlashFlood': 1,
    'Flood': 1,
    'HeavyRain': 0,
    'Unknown': 0,
    'random': 0
}


df_stats['label_yhat'] = df_stats['event'].map(label_map)


def rescale_with_minmax(x, lo, hi, eps=1e-8):
    rng = max(hi - lo, eps)
    return np.clip((x - lo) / rng, 0.0, 1.0)

def resize_frames_gpu(frames, target_size=(192, 192)):
    """
    frames: numpy array of shape [T, H, W]
    returns: resized array of shape [T, target_H, target_W]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(1).to(device)  # [T, 1, H, W]
    frames_resized = F.interpolate(frames_tensor, size=target_size, mode='bilinear', align_corners=False)  # [T, 1, 192, 192]
    return frames_resized.squeeze(1).cpu().numpy()  # back to [T, 192, 192]

def stack_modalities_from_list(file_list, use_vis=False):
    files = {'vis': None, 'ir069': None, 'ir107': None}

    for path in file_list:
        if '_vis_' in path:
            files['vis'] = path
        elif '_ir069_' in path:
            files['ir069'] = path
        elif '_ir107_' in path:
            files['ir107'] = path

    # Load IR channels
    SCALE = {"vis": 1.0e-4, "ir069": 1.0e-2, "ir107": 1.0e-2}

    ir069 = np.load(files['ir069'])['ir069'].astype(np.float32) * SCALE["ir069"]
    ir107 = np.load(files['ir107'])['ir107'].astype(np.float32) * SCALE["ir107"]
    ir069 = np.transpose(ir069, (2, 0, 1))
    ir107 = np.transpose(ir107, (2, 0, 1))

    # Optional: load and resize VIS channel
    if use_vis:
        vis = np.load(files['vis'])['vis'].astype(np.float32) * SCALE["vis"]
        vis = np.transpose(vis, (2, 0, 1))
        vis_resized = resize_frames_gpu(vis, target_size=(192, 192))  # [T, 192, 192]
        vis_rescaled = rescale_with_minmax(
            vis_resized,
            sensor_minmax['vis']['min'],
            sensor_minmax['vis']['max']
        )

    # Rescale IR channels
    ir069_rescaled = rescale_with_minmax(
        ir069,
        sensor_minmax['ir069']['min'],
        sensor_minmax['ir069']['max']
    )
    ir107_rescaled = rescale_with_minmax(
        ir107,
        sensor_minmax['ir107']['min'],
        sensor_minmax['ir107']['max']
    )

    # Check frame length match
    lengths = [ir069_rescaled.shape[0], ir107_rescaled.shape[0]]
    if use_vis:
        lengths.append(vis_rescaled.shape[0])
    if len(set(lengths)) != 1:
        raise ValueError(f"Frame mismatch across channels: {lengths}")

    # Stack channels
    if use_vis:
        stacked = np.stack([vis_rescaled, ir069_rescaled, ir107_rescaled], axis=-1)
    else:
        stacked = np.stack([ir069_rescaled, ir107_rescaled], axis=-1)

    return stacked

# # Create a save directory
save_dir = 'npz_files_stack_3ch/'
os.makedirs(save_dir, exist_ok=True)

start_time = time.time()
metadata = []
skipped = 0
processed = 0

# Example: grouped_files = {'S773562': ['/path/vis.npz', '/path/ir069.npz', '/path/ir107.npz'], ...}
# df_stats must contain 'event_id' and 'label_yhat' columns

# 👇 FIXED loop
for event_id, event_channels in grouped_files.items():
    paths = {'vis': None, 'ir069': None, 'ir107': None}
    for path in event_channels:
        if '_vis_' in path:
            paths['vis'] = path
        elif '_ir069_' in path:
            paths['ir069'] = path
        elif '_ir107_' in path:
            paths['ir107'] = path

    # Set this to True if you want to use VIS
    use_vis = True

    # Check required files
    required_keys = ['ir069', 'ir107'] + (['vis'] if use_vis else [])
    if not all(paths[k] for k in required_keys):
        skipped += 1
        continue

    try:
        sequence = stack_modalities_from_list(event_channels, use_vis=use_vis)

        filename = f"{event_id}.npz"
        path = os.path.join(save_dir, filename)
        np.savez_compressed(path, data=sequence)

        label = df_stats[df_stats['event_id'] == event_id]['label_yhat'].values[0]
        metadata.append((filename, label))

        processed += 1

    except Exception as e:
        print(f"⚠️ Skipping {event_id} due to error: {e}")
        skipped += 1

end_time = time.time()

print(f"Done. Processed: {processed}, Skipped: {skipped}, Time taken: {end_time - start_time:.2f} seconds")
pd.DataFrame(metadata, columns=["filename", "label"]).to_csv(f"{save_dir}/labels.csv", index=False)
