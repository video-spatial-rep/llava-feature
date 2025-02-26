import os
import torch

# Directory where the features are saved
feature_folder = "/home/rilyn/LLaVA-Video-feature/scannet_videos_128f/scene0000_00_128f.mp4/feature_folder"  # update this path as needed

# Iterate over all .pt files in the folder
for file_name in os.listdir(feature_folder):
    if file_name.endswith(".pt"):
        file_path = os.path.join(feature_folder, file_name)
        feature = torch.load(file_path)
        print(f"{file_name}: {feature.shape}")
