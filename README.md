# 3D Human Pose Estimation - AI Core

This repository contains the standalone Core Machine Learning pipeline for a contactless 3D Human Skeletal tracking application. The model processes multi-channel wireless signal arrays from an ESP32 sensor to accurately map spatial coordinates.

## 📊 Core Features & Training Analytics
- **Multi-Antenna Processing:** Custom Convolutional Neural Network (CNN) architecture configured with 3 input channels to natively handle 3-antenna ESP32 Wi-Fi CSI (Channel State Information) signals.
- **Robust Data Cleaning:** Integrated a custom hardware fault protection layer (`np.nan_to_num`) and Min-Max Normalization to handle corrupt sensor data segments.
- **Training Performance:** Optimization error dropped from an initial `0.2094` down to **`0.0097 MSE Loss`** over 5 epochs.
- **Validation Score:** Achieved an exceptional **`0.0092 Average Spatial Variance Error`** on completely unseen testing frames.

## 📁 Repository Contents
- `network.py`: The core deep learning layer architecture blueprint.
- `train.py`: The optimization script used to calculate the weighted tensor layers.
- `test.py`: Evaluates tracking precision and model stability against hidden evaluation data.
- `export_onnx.py`: Script to convert the Python tensor graph into a cross-platform format.
- `pose_tracker.onnx`: The fully trained, standalone mobile AI brain asset ready for frontend deployment.

## 🗺️ Future Roadmap
- [x] Phase 1: Machine Learning Core Selection & Training (Completed)
- [ ] Phase 2: Deno Backend API Integration
- [ ] Phase 3: Flutter Cross-Platform Frontend UI Development

