# 🚀 Quick Start Guide - USL System

## 5-Minute Setup

### Step 1: Download Models from Kaggle
```bash
# From your Kaggle notebook, download:
# - sign_recognition_model.pth
# - usl_screening_model.pth
# - sign_vocabulary.json

# Create directory and place files
mkdir -p usl_models
# Copy downloaded files to usl_models/
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Streamlit App
```bash
streamlit run app_updated.py
```

### Step 4: Upload & Process Video
1. Open browser to `http://localhost:8501`
2. Click "Upload Video"
3. Click "Process Video"
4. View results!

---

## Usage Examples

### Example 1: Process a Video File
```python
from usl_inference import USLInferencePipeline

# Initialize
pipeline = USLInferencePipeline(
    sign_model_path='./usl_models/sign_recognition_model.pth',
    screening_model_path='./usl_models/usl_screening_model.pth',
    sign_vocab_path='./usl_models/sign_vocabulary.json'
)

# Process video
result = pipeline.process_video('patient_video.mp4')

# Print results
print("Recognized signs:", result['signs']['sign_names'])
print("Screening slot:", result['screening']['screening_slot'])
print("Response:", result['screening']['response'])
print("Confidence:", result['screening']['confidence'])
```

### Example 2: Recognize Signs Only
```python
# Extract pose
pose = pipeline.extract_pose_from_video('video.mp4')

# Recognize signs
signs = pipeline.recognize_signs(pose)
print(signs['sign_names'])  # ['fever', 'three', 'day']
```

### Example 3: Classify Screening
```python
# Extract pose
pose = pipeline.extract_pose_from_video('video.mp4')

# Classify
screening = pipeline.classify_screening(pose)
print(f"Slot: {screening['screening_slot']}")
print(f"Response: {screening['response']}")
print(f"Confidence: {screening['confidence']:.2%}")
```

---

## What Each Model Does

### Sign Recognition Model
**Input**: Video of sign language
**Output**: Sequence of recognized signs

```
Video → Pose Extraction → LSTM+Attention → CTC Decoder → Signs
Example: "fever three day" (3 days of fever)
```

**Recognizes**: 45 signs including:
- Symptoms: fever, cough, blood, pain, diarrhea, rash, etc.
- Numbers: zero, one, two, three, four, five, six, seven, eight, nine, ten
- Temporal: day, week, month, today, yesterday, ago
- Responses: yes, no, maybe
- Severity: mild, moderate, severe, emergency

### Screening Classifier Model
**Input**: Video of sign language
**Output**: Screening slot + Response + Confidence

```
Video → Pose Extraction → LSTM+Attention → Classification → Result
Example: Slot="fever", Response="yes", Confidence=0.92
```

**Classifies**: 10 screening slots
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

---

## File Structure

```
project/
├── usl_models/                          # Model files
│   ├── sign_recognition_model.pth       # Sign recognition weights
│   ├── usl_screening_model.pth          # Screening classifier weights
│   ├── sign_vocabulary.json             # Sign vocabulary
│   └── training_results.png             # Training curves
│
├── usl_inference.py                     # Inference pipeline
├── app_updated.py                       # Streamlit app
├── app.py                               # Original app
├── complete_usl_system.py               # Original system
│
├── requirements.txt                     # Dependencies
├── README_MODELS.md                     # Full documentation
├── TRAINING_COMPLETE.md                 # Training summary
└── QUICK_START.md                       # This file
```

---

## Common Tasks

### Task 1: Process a Single Video
```python
from usl_inference import USLInferencePipeline

pipeline = USLInferencePipeline(
    './usl_models/sign_recognition_model.pth',
    './usl_models/usl_screening_model.pth',
    './usl_models/sign_vocabulary.json'
)

result = pipeline.process_video('video.mp4')
print(result)
```

### Task 2: Batch Process Multiple Videos
```python
import glob

videos = glob.glob('videos/*.mp4')
results = []

for video in videos:
    result = pipeline.process_video(video)
    results.append(result)
    print(f"Processed {video}")

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Task 3: Extract Just the Signs
```python
pose = pipeline.extract_pose_from_video('video.mp4')
signs = pipeline.recognize_signs(pose)

print("Signs recognized:")
for i, sign in enumerate(signs['sign_names'], 1):
    print(f"  {i}. {sign}")
```

### Task 4: Get Screening Classification
```python
pose = pipeline.extract_pose_from_video('video.mp4')
screening = pipeline.classify_screening(pose)

print(f"Question: {screening['screening_slot']}")
print(f"Answer: {screening['response']}")
print(f"Confidence: {screening['confidence']:.1%}")
```

### Task 5: Generate FHIR Bundle
```python
import json
from datetime import datetime

result = pipeline.process_video('video.mp4')
screening = result['screening']

fhir_bundle = {
    "resourceType": "Bundle",
    "type": "collection",
    "timestamp": datetime.now().isoformat(),
    "entry": [{
        "resource": {
            "resourceType": "Observation",
            "status": "final",
            "code": {"coding": [{"code": screening['screening_slot']}]},
            "valueString": screening['response'],
            "extension": [{
                "url": "http://medisign.ug/confidence",
                "valueDecimal": screening['confidence']
            }]
        }
    }]
}

with open('fhir_bundle.json', 'w') as f:
    json.dump(fhir_bundle, f, indent=2)
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### Problem: "FileNotFoundError: sign_recognition_model.pth"
**Solution**: Ensure models are in `./usl_models/`
```bash
ls -la usl_models/
# Should show: sign_recognition_model.pth, usl_screening_model.pth, sign_vocabulary.json
```

### Problem: "CUDA out of memory"
**Solution**: Use CPU instead
```python
pipeline = USLInferencePipeline(..., device='cpu')
```

### Problem: "Empty sign sequence"
**Solution**: This is normal for synthetic data. Models work better on real USL videos.

### Problem: "Streamlit app won't start"
**Solution**: Check port 8501 is available
```bash
streamlit run app_updated.py --server.port 8502
```

---

## Performance Tips

### Speed Up Inference
1. Use GPU: `device='cuda'`
2. Reduce sequence length: `max_frames=300`
3. Batch process: Process multiple videos together

### Improve Accuracy
1. Use real USL videos (not synthetic)
2. Ensure good lighting and camera angle
3. Keep signer in frame
4. Use high-quality video (720p+)

### Reduce Memory Usage
1. Use CPU: `device='cpu'`
2. Reduce batch size
3. Process shorter videos

---

## Model Specifications

### Sign Recognition Model
- **Input**: (batch, 500, 33, 3) - 500 frames, 33 joints, 3D coords
- **Output**: (batch, 500, 46) - CTC logits for 46 classes
- **Parameters**: 13.5M
- **Inference Time**: ~280ms per video
- **Memory**: ~2GB GPU / ~4GB CPU

### Screening Classifier Model
- **Input**: (batch, 150, 33, 3) - 150 frames, 33 joints, 3D coords
- **Output**: (batch, 10), (batch, 3), (batch, 1) - slot, response, confidence
- **Parameters**: 2.5M
- **Inference Time**: ~150ms per video
- **Memory**: ~1GB GPU / ~2GB CPU

---

## Next Steps

1. **Test on Real Data**: Collect USL videos and test models
2. **Validate Results**: Compare with human annotations
3. **Collect Feedback**: Get input from deaf community
4. **Fine-tune**: Train on real data for better accuracy
5. **Deploy**: Integrate with clinical systems

---

## Resources

- **Full Documentation**: See `README_MODELS.md`
- **Training Details**: See `TRAINING_COMPLETE.md`
- **Code**: See `usl_inference.py`
- **App**: See `app_updated.py`

---

## Support

For issues:
1. Check this guide
2. Read `README_MODELS.md`
3. Review code comments in `usl_inference.py`
4. Check Streamlit app logs

---

**Ready to go!** 🚀

Start with: `streamlit run app_updated.py`
