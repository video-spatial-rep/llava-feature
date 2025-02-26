import os
import torch
import torch.nn as nn
import numpy as np

# Define the MLP that compresses a row of shape (3584,) to (896,)
class RowWiseCompressor(nn.Module):
    def __init__(self, in_features=3584, hidden_features=2048, out_features=896):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
    
    def forward(self, x):
        # x is assumed to have shape (729, 3584)
        # Apply the MLP on the last dimension
        x = x.float()
        return self.mlp(x)

def load_and_sort_features(feature_folder):
    # Get list of all .pt files in the folder
    files = [f for f in os.listdir(feature_folder) if f.endswith(".pt")]
    # Extract the numerical index from the filename (assumes format: feature_{idx}.pt)
    files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return [os.path.join(feature_folder, f) for f in files]

def process_features(feature_folder):
    # Load sorted feature file paths
    feature_files = load_and_sort_features(feature_folder)
    
    # Verify we have a multiple of 4 feature files
    if len(feature_files) % 4 != 0:
        raise ValueError("The number of feature files is not a multiple of 4.")

    compressor = RowWiseCompressor()
    compressor.eval()  # set to evaluation mode
    all_compressed = []

    # Group features four-by-four
    for group_idx in range(0, len(feature_files), 4):
        group_files = feature_files[group_idx:group_idx+4]
        # Load each feature tensor; each should have shape (729, 896)
        group_tensors = [torch.load(f) for f in group_files]
        # Concatenate along the feature dimension (dim=1)
        # Resulting shape: (729, 896 * 4 = 3584)
        concatenated = torch.cat(group_tensors, dim=1)
        
        # Apply the MLP row-wise: compressor expects input of shape (729, 3584)
        with torch.no_grad():
            compressed = compressor(concatenated)  # shape: (729, 896)
        all_compressed.append(compressed)

    # Stack all compressed tensors: final shape: (num_groups, 729, 896)
    final_tensor = torch.stack(all_compressed, dim=0)
    print("Final tensor shape:", final_tensor.shape)
    return final_tensor

def process_all_ids(input_base, output_base):
    """
    For each subdirectory (id) in input_base:
      - Reads features from: {input_base}/{id}/feature_files
      - Processes and compresses the features.
      - Saves the final tensor as: {output_base}/{id}.llava.npy
    """
    # Iterate over each subdirectory in the input_base
    for id_dir in os.listdir(input_base):
        subdir_path = os.path.join(input_base, id_dir)
        # Ensure that it is a directory
        if os.path.isdir(subdir_path):
            feature_folder = os.path.join(subdir_path, "feature_folder")
            if not os.path.exists(feature_folder):
                print(f"Skipping {id_dir}: 'feature_files' folder not found.")
                continue
            try:
                final_tensor = process_features(feature_folder)
            except Exception as e:
                print(f"Error processing {id_dir}: {e}")
                continue
       
            # Prepare the output file path
            output_file = os.path.join(output_base, f"{id_dir}.llava.npy")
            # Save the final tensor as a NPY file
            np.save(output_file, final_tensor.cpu().numpy())
            print(f"Saved compressed features for {id_dir} to {output_file}")

# Usage example:
input_base = "/home/rilyn/LLaVA-Video-feature/scannet_videos_128f"
output_base = "/nas/spatial/videos/scannet_videos_128f"
process_all_ids(input_base, output_base)
