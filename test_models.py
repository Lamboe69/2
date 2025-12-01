#!/usr/bin/env python3
"""Test script to verify USL models load correctly"""

try:
    from usl_inference import USLInferencePipeline
    print("✅ USL inference module imported successfully")

    # Try to load models
    pipeline = USLInferencePipeline(
        sign_model_path='./usl_models/sign_recognition_model.pth',
        screening_model_path='./usl_models/usl_screening_model.pth',
        sign_vocab_path='./usl_models/sign_vocabulary.json',
        device='cpu'
    )

    print("✅ Models loaded successfully!")
    print(f"Sign vocabulary size: {len(pipeline.sign_vocab)}")
    print(f"Screening slots: {len(pipeline.screening_slots)}")
    print(f"Device: {pipeline.device}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
