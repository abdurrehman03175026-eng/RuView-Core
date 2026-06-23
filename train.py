import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from network import MMFiPoseDataset, CSIPoseNet

def run_training():
    # Adjusted exactly to your true folder path configuration
    DATASET_ROOT = r"D:\ruview_dataset\MMFi_Extracted"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_files = glob.glob(os.path.join(DATASET_ROOT, "MMFi_Dataset", "E*", "S*", "A*", "wifi-csi", "*.mat"))
    if not all_files:
        print("ERROR: No .mat files found. Double-check your DATASET_ROOT folder variable!")
        return
        
    random.seed(42)
    random.shuffle(all_files)
    
    # 80/20 Dataset Split
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    
    with open("test_files_list.txt", "w") as f:
        for item in all_files[split_idx:]:
            f.write(f"{item}\n")
            
    train_dataset = MMFiPoseDataset(DATASET_ROOT, train_files)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = CSIPoseNet().to(device)
    criterion = nn.MSELoss() # Changed to MSE Loss for continuous coordinate prediction tracking
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Tracking 3D Coordinates over {len(train_files)} frames on device: {device}")
    
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/5] - Tracking Spatial Error (MSE Loss): {running_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), "frozen_weights.pth")
    print("\nSUCCESS: 'frozen_weights.pth' 3D tracking tensor layers generated!")

if __name__ == "__main__":
    run_training()
