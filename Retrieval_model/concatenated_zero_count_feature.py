import numpy as np
import os
from tqdm import tqdm  # For progress bar

# Path where per-block TZCPB features are saved
folder_path = 'C:/Users/AI/Desktop/FinalEViT/data/features/zero_count_feature/'

# Output file path
output_path = 'zero_count_feature.npy'

# Load all .npy files, track max block dimension
arrays = []
max_dim = 0
verbose = True

for x in tqdm(range(100), desc="Loading TZCPB", ncols=100):
    for y in range(x * 100 + 1, (x + 1) * 100 + 1):
        file_path = os.path.join(folder_path, f'{x}_{y}.npy')

        if os.path.exists(file_path):
            try:
                arr = np.load(file_path)
                if arr.ndim == 2:
                    arrays.append(arr)
                    max_dim = max(max_dim, arr.shape[0])
                    if verbose:
                        print(f"Loaded {file_path} with shape {arr.shape}")
                else:
                    print(f"Skipped {file_path}: Expected 2D, got {arr.ndim}D")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: {file_path} not found.")

# Pad all arrays to the same number of blocks (max_dim)
padded_arrays = []
for arr in arrays:
    if arr.shape[0] < max_dim:
        pad_len = max_dim - arr.shape[0]
        pad = ((0, pad_len), (0, 0))  # Pad rows (block count), not columns
        arr = np.pad(arr, pad, mode='constant', constant_values=0)
        if verbose:
            print(f"Padded to {arr.shape}")
    padded_arrays.append(arr)

# Stack into a single array
if padded_arrays:
    try:
        concatenated_array = np.stack(padded_arrays, axis=0)
        print(f"Final shape: {concatenated_array.shape}")
        np.save(output_path, concatenated_array)
        print(f"✅ TZCPB feature saved as: {output_path}")
    except Exception as e:
        print(f"❌ Error during stacking: {e}")
else:
    print("❌ No TZCPB arrays to process.")
