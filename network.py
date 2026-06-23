import os
import glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MMFiPoseDataset(Dataset):
    def __init__(self, base_path, file_list):
        self.file_paths = file_list
        self.gt_cache = {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        path_segments = file_path.split(os.sep)
        
        filename = path_segments[-1]
        frame_idx = int(''.join(filter(str.isdigit, filename))) - 1
        
        parent_dir = os.path.dirname(file_path)
        action_dir = os.path.dirname(parent_dir)
        gt_path = os.path.join(action_dir, "ground_truth.npy")
        
        if gt_path not in self.gt_cache:
            self.gt_cache[gt_path] = np.load(gt_path, allow_pickle=True)
        
        gt_data = self.gt_cache[gt_path]
        
        if frame_idx >= len(gt_data):
            frame_idx = len(gt_data) - 1
        target_coords = gt_data[frame_idx]
        
        mat_data = sio.loadmat(file_path)
        
        possible_keys = ['CSIamp', 'csi', 'csi_matrix', 'CSI', 'CSI_matrix']
        csi_key = None
        for key in possible_keys:
            if key in mat_data:
                csi_key = key
                break
                
        if csi_key is None:
            raise KeyError(f"Could not find a valid CSI matrix key in file: {filename}.")
            
        raw_csi = mat_data[csi_key].astype(np.float32)
        
        # --- FIXED: DATA CLEANING LAYER TO REMOVE CORRUPT HARDWARE VALUES ---
        # Replaces any broken NaN or Inf values with 0.0 so the math never explodes
        raw_csi = np.nan_to_num(raw_csi, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- NORMALIZATION LAYER ---
        csi_min = raw_csi.min()
        csi_max = raw_csi.max()
        if csi_max > csi_min:
            normalized_csi = (raw_csi - csi_min) / (csi_max - csi_min)
        else:
            normalized_csi = np.zeros_like(raw_csi) # Safe fallback for empty frames
        
        target_flat = target_coords.flatten()
        
        return torch.tensor(normalized_csi, dtype=torch.float32), torch.tensor(target_flat, dtype=torch.float32)

class CSIPoseNet(nn.Module):
    def __init__(self, output_dim=51):
        super(CSIPoseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.regressor(self.features(x))
