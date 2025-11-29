# USL System - Training Summary

## Project Overview

**Graph-Reasoned Large Vision Models for Ugandan Sign Language Translation in Infectious Disease Screening**

This project implements a complete machine learning pipeline for training a model to:
- Classify screening slots (10 categories: fever, cough, diarrhea, etc.)
- Predict patient responses (yes/no/unknown)
- Estimate confidence scores

## Training Results

### Final Metrics
- **Test Slot Accuracy**: 100%
- **Test Response Accuracy**: 100%
- **Test Loss**: 0.8149
- **Model Parameters**: 2,487,374

### Training Configuration
- **Dataset**: 500 training samples, 100 test samples
- **Sequence Length**: 150 frames per video
- **Joints**: 33 (MediaPipe pose landmarks)
- **Batch Size**: 32
- **Learning Rate**: 0.0005
- **Epochs**: 75 (early stopped)
- **Device**: CUDA (GPU)

### Model Architecture
```
ImprovedPoseModel(
  - Joint Embedding: 3D → 128D
  - LSTM: 2 layers, 128 hidden units
  - Attention: Multi-head (4 heads)
  - Classification Heads:
    * Slot Classifier: 10 classes
    * Response Classifier: 3 classes
    * Confidence Head: [0, 1]
)
```

### Key Features
1. **Regularization**:
   - Dropout (0.3-0.4)
   - Batch Normalization
   - Label Smoothing (0.1)
   - Weight Decay (0.01)

2. **Optimization**:
   - AdamW optimizer
   - ReduceLROnPlateau scheduler
   - Gradient clipping (max_norm=1.0)
   - Early stopping (patience=15)

3. **Architecture**:
   - Temporal LSTM for sequence modeling
   - Multi-head attention for temporal focus
   - Separate classification heads for multi-task learning

## Data

### Synthetic Data Generation
- **Purpose**: Proof-of-concept training with structured patterns
- **Format**: (batch, seq_len=150, joints=33, coords=3)
- **Labels**: Screening slots + responses

### Real Data Integration
To use real data:
1. Extract pose sequences using MediaPipe
2. Annotate with screening slot labels
3. Load using custom DataLoader
4. Train with same pipeline

## Model Checkpoints

Saved files:
- `usl_models/best_model.pth` - Best model weights
- `usl_models/usl_screening_model.pth` - Final model with config
- `usl_models/training_results.png` - Training curves

## Usage

### Training
```python
from usl_training_pipeline import train_model, Config

config = Config()
model, config = train_model(config)
```

### Inference
```python
import torch
import numpy as np

# Load model
checkpoint = torch.load('usl_models/usl_screening_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input
pose_sequence = torch.FloatTensor(np.random.randn(1, 150, 33, 3))

# Predict
with torch.no_grad():
    slot_logits, response_logits, confidence = model(pose_sequence)
    slot_pred = torch.argmax(slot_logits, dim=1).item()
    response_pred = torch.argmax(response_logits, dim=1).item()
```

## Next Steps

1. **Collect Real USL Data**:
   - Record videos of native signers
   - Extract poses using MediaPipe
   - Annotate with screening labels

2. **Fine-tune Model**:
   - Load pre-trained weights
   - Train on real data
   - Adjust hyperparameters

3. **Deploy**:
   - Convert to ONNX/TorchScript
   - Optimize for mobile (INT8 quantization)
   - Integrate with Streamlit app

4. **Evaluate**:
   - Test on real clinical data
   - Measure latency
   - Assess accuracy on diverse signers

## References

- MediaPipe Pose: https://mediapipe.dev/
- PyTorch: https://pytorch.org/
- Attention Mechanisms: https://arxiv.org/abs/1706.03762
