import os
import torch
import torch.nn as nn # FIXED: Added the missing neural network module import
from torch.utils.data import DataLoader
from network import MMFiPoseDataset, CSIPoseNet

def run_testing():
    DATASET_ROOT = r"D:\ruview_dataset\MMFi_Extracted"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists("test_files_list.txt"):
        print("Error: Run train.py first to split target sequences.")
        return
        
    with open("test_files_list.txt", "r") as f:
        test_files = [line.strip() for line in f.readlines() if line.strip()]
        
    test_dataset = MMFiPoseDataset(DATASET_ROOT, test_files)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = CSIPoseNet().to(device)
    model.load_state_dict(torch.load("frozen_weights.pth", map_location=device))
    model.eval()
    
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    print(f"Evaluating model precision on {len(test_files)} testing sequences...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
    avg_error = total_loss / len(test_loader)
    print(f"\n--- MODEL COORDINATE PRECISION RESULTS ---")
    print(f"Average Spatial Variance Error (MSE): {avg_error:.4f}")
    print("Lower error values indicate highly accurate 3D structural tracking capabilities.")

if __name__ == "__main__":
    run_testing()
