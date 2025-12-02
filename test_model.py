#!/usr/bin/env python3
"""
Test the real-trained model
"""

from usl_inference_real import USLInferencePipelineReal
import os

print("=" * 80)
print("TESTING REAL-TRAINED MODEL")
print("=" * 80)

# Initialize pipeline
print("\n🔧 Initializing pipeline...")
try:
    pipeline = USLInferencePipelineReal(
        model_path='./usl_models/real_video_model.pth',
        device='cpu'  # Use CPU for testing
    )
    print("✅ Pipeline initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing pipeline: {e}")
    exit(1)

# Find test videos
print("\n📁 Looking for test videos...")
test_videos = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov')):
            test_videos.append(os.path.join(root, file))

if len(test_videos) == 0:
    print("❌ No video files found!")
    print("   Please provide a video file to test")
    print("   Supported formats: .mp4, .avi, .mov")
    exit(1)

print(f"✅ Found {len(test_videos)} video file(s)")

# Test on first video
print("\n" + "=" * 80)
print("TESTING ON VIDEO")
print("=" * 80)

video_path = test_videos[0]
print(f"\n📹 Testing on: {video_path}")

try:
    result = pipeline.process_video(video_path)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\n✅ Classification Result:")
    print(f"   Class: {result['class']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Frames Extracted: {result.get('frames_extracted', 'N/A')}")
    
    if 'all_probs' in result:
        print(f"\n📊 All Probabilities:")
        for class_name, prob in result['all_probs'].items():
            print(f"   {class_name}: {prob:.2%}")
    
    if 'error' in result:
        print(f"\n⚠️  Error: {result['error']}")
    
    print("\n✅ TEST SUCCESSFUL!")

except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
