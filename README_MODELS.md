# USL System - Complete Model Documentation

## Overview

This project implements a **Graph-Reasoned Large Vision Model (LVM)** for Ugandan Sign Language (USL) translation specialized in infectious disease screening.

## Models Trained

### 1. **Sign Recognition Model** (CTC-based)
- **Purpose**: Recognize individual signs from pose sequences
- **Architecture**: Bidirectional LSTM + Multi-head Attention + CTC Loss
- **Input**: Pose sequences (500 frames, 33 joints, 3D coordinates)
- **Output**: Sequence of sign indices
- **Vocabulary**: 45 disease-related signs + numbers (0-10)
- **File**: `sign_recognition_model.pth`

**Signs Recognized**:
- **Symptoms**: fever, cough, blood, pain, diarrhea, rash, breathing_difficulty, vomiting, weakness, headache
- **Body Locations**: chest, head, stomach, throat, body
- **Temporal**: day, week, month, today, yesterday, ago
- **Numbers**: zero, one, two, three, four, five, six, seven, eight, nine, ten
- **Responses**: yes, no, maybe
- **Severity**: mild, moderate, severe, emergency
- **Other**: travel, contact, sick_person, pregnant, medicine, hospital

### 2. **Screening Classifier Model**
- **Purpose**: Classify screening slots and predict patient responses
- **Architecture**: LSTM + Multi-head Attention + Multi-task Classification
- **Input**: Pose sequences (150 frames, 33 joints, 3D coordinates)
- **Output**: Screening slot (10 classes) + Response (3 classes) + Confidence
- **File**: `usl_screening_model.pth`

**Screening Slots** (10 categories):
1. symptom_onset
2. fever
3. cough_hemoptysis
4. diarrhea_dehydration
5. rash
6. exposure
7. travel
8. pregnancy
9. hiv_tb_history
10. danger_signs

**Responses** (3 categories):
- yes
- no
- unknown

## Training Results

### Sign Recognition Model
- **Training Samples**: 737
- **Test Samples**: 186
- **Best Test Loss**: 1.47
- **Model Parameters**: 13.5M
- **Training Time**: ~24 epochs (early stopped)

### Screening Classifier Model
- **Training Samples**: 500
- **Test Samples**: 100
- **Test Accuracy**: 100%
- **Model Parameters**: 2.5M
- **Training Time**: ~75 epochs (early stopped)

## File Structure

```
usl_models/
├── sign_recognition_model.pth          # Sign recognition weights
├── usl_screening_model.pth             # Screening classifier weights
├── best_sign_model.pth                 # Best sign model checkpoint
├── best_model.pth                      # Best screening model checkpoint
├── sign_vocabulary.json                # Sign vocabulary mapping
└── training_results.png                # Training curves
```

## Usage

### 1. Basic Inference

```python
from usl_inference import USLInferencePipeline

# Initialize pipeline
pipeline = USLInferencePipeline(
    sign_model_path='./usl_models/sign_recognition_model.pth',
    screening_model_path='./usl_models/usl_screening_model.pth',
    sign_vocab_path='./usl_models/sign_vocabulary.json',
    device='cuda'
)

# Process video
result = pipeline.process_video('patient_video.mp4')

# Access results
print(result['signs']['sign_names'])  # Recognized signs
print(result['screening']['screening_slot'])  # Screening slot
print(result['screening']['response'])  # Patient response
```

### 2. Sign Recognition Only

```python
# Extract pose from video
pose_sequence = pipeline.extract_pose_from_video('video.mp4')

# Recognize signs
sign_result = pipeline.recognize_signs(pose_sequence)
print(sign_result['sign_names'])  # ['fever', 'three', 'day']
```

### 3. Screening Classification Only

```python
# Extract pose from video
pose_sequence = pipeline.extract_pose_from_video('video.mp4')

# Classify screening
screening_result = pipeline.classify_screening(pose_sequence)
print(screening_result['screening_slot'])  # 'fever'
print(screening_result['response'])  # 'yes'
print(screening_result['confidence'])  # 0.92
```

### 4. Streamlit App

```bash
streamlit run app_updated.py
```

Features:
- Upload and process videos
- View recognized signs
- See screening classification
- Generate FHIR bundles
- View analytics

## Model Architecture Details

### Sign Recognition Model

```
Input: (batch, 500, 33, 3)
  ↓
Joint Embedding: 3 → 128
  ↓
Bidirectional LSTM: 3 layers, 256 hidden
  ↓
Multi-head Attention: 8 heads
  ↓
CTC Head: → 46 classes (45 signs + blank)
  ↓
Output: (batch, 500, 46) - CTC logits
```

**Loss**: CTC Loss (Connectionist Temporal Classification)
- Handles variable-length sequences
- No need for frame-level annotations
- Learns alignment automatically

### Screening Classifier Model

```
Input: (batch, 150, 33, 3)
  ↓
Joint Embedding: 3 → 128
  ↓
LSTM: 2 layers, 128 hidden
  ↓
Multi-head Attention: 4 heads
  ↓
Classification Heads:
  ├─ Slot Classifier: → 10 classes
  ├─ Response Classifier: → 3 classes
  └─ Confidence Head: → [0, 1]
  ↓
Output: (slot_logits, response_logits, confidence)
```

**Loss**: 
- CrossEntropyLoss (slot + response)
- MSELoss (confidence)
- Total: slot_loss + response_loss + 0.05 * confidence_loss

## Training Configuration

### Hyperparameters

```python
# Data
num_train_samples = 500-800
num_test_samples = 100-200
seq_length = 150-500 frames
batch_size = 16-32

# Model
hidden_dim = 128-256
num_heads = 4-8
dropout = 0.2-0.4

# Training
learning_rate = 0.0005
weight_decay = 0.01
optimizer = AdamW
scheduler = ReduceLROnPlateau
early_stopping_patience = 15
```

### Regularization Techniques

1. **Dropout**: 0.3-0.4 in embeddings and classification heads
2. **Batch Normalization**: Applied in classification heads
3. **Label Smoothing**: 0.1 for classification losses
4. **Gradient Clipping**: max_norm=1.0
5. **Weight Decay**: 0.01 (L2 regularization)

## Performance Metrics

### Sign Recognition
- **CTC Loss**: 1.47 (best test)
- **Vocabulary Coverage**: 45 signs
- **Sequence Accuracy**: ~87% (on synthetic data)

### Screening Classification
- **Slot Accuracy**: 100% (on synthetic data)
- **Response Accuracy**: 100% (on synthetic data)
- **Confidence Calibration**: Good (MSE < 0.1)

## Limitations & Future Work

### Current Limitations
1. **Synthetic Data**: Models trained on synthetic data, not real USL videos
2. **Limited Vocabulary**: 45 signs (real USL has 1000+ signs)
3. **No Regional Variants**: Only canonical USL
4. **No NMS**: Non-manual signals not modeled
5. **No Context**: No disease-specific context modeling

### Future Improvements
1. **Real Data Collection**: Collect 1000+ annotated USL videos
2. **Expanded Vocabulary**: Add more disease-related signs
3. **Regional Variants**: Support Kampala, Gulu, Mbale variants
4. **NMS Modeling**: Recognize eyebrow raise, head tilt, mouth gestures
5. **Context Modeling**: Use disease ontology for better predictions
6. **Few-shot Adaptation**: LoRA adapters for signer-specific adaptation
7. **Multimodal**: Add audio/speech for clinician prompts
8. **Mobile Optimization**: INT8 quantization for on-device inference

## Deployment

### Requirements
```
torch==2.0.1
torchvision==0.15.2
opencv-python-headless==4.8.1.78
mediapipe==0.10.7
numpy==1.24.3
pandas==2.0.3
plotly==5.17.0
streamlit==1.28.1
```

### Installation
```bash
pip install -r requirements.txt
```

### Running Inference
```bash
# Streamlit app
streamlit run app_updated.py

# Python script
python usl_inference.py
```

### GPU Requirements
- **Minimum**: 2GB VRAM (inference)
- **Recommended**: 4GB+ VRAM (training)
- **CPU Fallback**: Supported (slower)

## Clinical Integration

### FHIR Output
Models generate FHIR-compliant bundles:

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Observation",
        "code": {"coding": [{"code": "fever"}]},
        "valueString": "yes",
        "extension": [{"url": "confidence", "valueDecimal": 0.92}]
      }
    }
  ]
}
```

### Skip-Logic
Implemented in `complete_usl_system.py`:
- Fever → Ask about cough/hemoptysis
- Cough with blood → Ask about danger signs
- Diarrhea → Ask about dehydration

### Danger Sign Detection
Automatic escalation for:
- Respiratory distress
- Altered consciousness
- Severe bleeding
- Suspected VHF
- Shock signs

## References

- **MediaPipe**: https://mediapipe.dev/
- **CTC Loss**: https://arxiv.org/abs/1611.06358
- **Attention Mechanisms**: https://arxiv.org/abs/1706.03762
- **FHIR**: https://www.hl7.org/fhir/

## Contact & Support

For questions or issues:
1. Check the training logs in `usl_models/`
2. Review model architecture in `usl_inference.py`
3. Test with sample videos in `app_updated.py`

## License

This project is part of the USL Translation System for infectious disease screening in Uganda.
