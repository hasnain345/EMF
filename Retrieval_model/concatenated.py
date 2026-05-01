import numpy as np
import os

# Define the path to the folder where the .npy files are located
folder_path = 'C:/Users/AI/Desktop/FinalEViT/data/features/difffeature_matrix/'

# Initialize an empty list to hold all the loaded arrays
arrays = []

# Variable to store the maximum second dimension size
max_dim = 0

# Optional verbosity flag
verbose = True

# Iterate through the range of file indices (from 0_1.npy to 99_10000.npy)
for x in range(100):  # First part of the file name ranges from 0 to 99
    # For each x, the second part (y) goes from x*100 + 1 to (x+1)*100
    for y in range(x * 100 + 1, (x + 1) * 100 + 1):
        # Construct the file path for each .npy file
        file_path = os.path.join(folder_path, f'{x}_{y}.npy')
        
        # Check if the file exists before loading it
        if os.path.exists(file_path):
            # Load the .npy file
            arr = np.load(file_path)
            if verbose:
                print(f'Loaded {file_path} with shape {arr.shape}')
            
            # Update max_dim to the largest second dimension size encountered
            max_dim = max(max_dim, arr.shape[1])
            
            arrays.append(arr)
        else:
            if verbose:
                print(f"Warning: {file_path} does not exist")

# After finding the maximum second dimension size, pad arrays to this size
padded_arrays = []

for arr in arrays:
    if arr.shape[1] < max_dim:
        # Calculate padding for the second dimension (dimension 1)
        padding = ((0, 0), (0, max_dim - arr.shape[1]), (0, 0))  # Pad only the second dimension
        arr = np.pad(arr, padding, mode='constant', constant_values=0)
        if verbose:
            print(f'Padding applied, new shape: {arr.shape}')
    
    padded_arrays.append(arr)

# Check if there are arrays to concatenate
if padded_arrays:
    try:
        # Concatenate all the arrays along the first axis (axis=0)
        concatenated_array = np.concatenate(padded_arrays, axis=0)

        # Print the shape of the concatenated array
        print(f"Shape of concatenated array: {concatenated_array.shape}")

        # Save the concatenated array into a single .npy file
        output_path = 'difffeature_matrix.npy'
        np.save(output_path, concatenated_array)

        print(f"Concatenation complete and saved as '{output_path}'")
    except ValueError as e:
        print(f"Error during concatenation: {e}")
else:
    print("No arrays to concatenate. Please check the files.")
