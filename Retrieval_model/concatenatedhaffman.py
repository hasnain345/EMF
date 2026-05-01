import numpy as np
import os
from tqdm import tqdm  # Optional: for progress bar, install tqdm via pip if needed

# Define the path to the folder where the .npy files are located
folder_path = 'C:/Users/AI/Desktop/FinalEViT/data/features/huffman_feature/'

# Initialize an empty list to hold all the loaded arrays
arrays = []

# Optional verbosity flag for more control over output
verbose = True

# Iterate through the range 0 to 99 (first part of the file name)
for x in tqdm(range(100), desc="Loading files", ncols=100):  # Using tqdm for progress bar
    # For each x, the second part (y) goes from x*100 + 1 to (x+1)*100
    for y in range(x * 100 + 1, (x + 1) * 100 + 1):
        # Construct the file path for each .npy file
        file_path = os.path.join(folder_path, f'{x}_{y}.npy')
        
        # Check if the file exists before loading it
        if os.path.exists(file_path):
            try:
                # Load the .npy file
                arr = np.load(file_path)
                
                # Check if the array shape is consistent with the first loaded array
                if arrays and arr.shape != arrays[0].shape:
                    print(f"Warning: Shape mismatch for {file_path}, skipping this file.")
                    continue  # Skip files with inconsistent shapes
                
                arrays.append(arr)
                if verbose:
                    print(f'Loaded {file_path} with shape {arr.shape}')  # Print the shape of each array
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: {file_path} does not exist")

# Check if there are arrays to concatenate
if arrays:
    try:
        # Concatenate all the arrays along the first axis (axis=0)
        concatenated_array = np.concatenate(arrays, axis=0)

        # Save the concatenated array into a single .npy file
        output_path = 'huffman_feature.npy'
        np.save(output_path, concatenated_array)

        print(f"Concatenation complete and saved as '{output_path}'")
    except ValueError as e:
        print(f"Error during concatenation: {e}")
else:
    print("No arrays to concatenate. Please check the files.")
