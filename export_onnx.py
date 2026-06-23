import torch
from network import CSIPoseNet

def export_to_onnx():
    device = torch.device("cpu")
    
    # 1. Initialize the network structure and load your brilliant new weights
    model = CSIPoseNet()
    model.load_state_dict(torch.load("frozen_weights.pth", map_location=device))
    model.eval()
    
    # 2. Create a dummy input matrix mimicking your 3-antenna ESP32 data format [Batch, Antennas, Subcarriers, Samples]
    dummy_input = torch.randn(1, 3, 114, 10)
    
    # 3. Export the tensor graph to a clean mobile asset file
    output_filename = "pose_tracker.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        output_filename, 
        export_params=True, 
        opset_version=12,
        input_names=['wi_fi_signals'], 
        output_names=['predicted_3d_joints']
    )
    print(f"SUCCESS: Mobile asset model generated and saved as '{output_filename}'!")

if __name__ == "__main__":
    export_to_onnx()

