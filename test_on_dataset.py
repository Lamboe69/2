#!/usr/bin/env python3
"""
Test the real-trained model on your downloaded ISL dataset
"""

import os
from pathlib import Path
from usl_inference_real import USLInferencePipelineReal
import json

print("=" * 80)
print("TESTING MODEL ON ISL DATASET")
print("=" * 80)

# ============================================================================
# 1. FIND DATASET LOCATION
# ============================================================================

print("\n🔍 Looking for ISL dataset...")

# Common locations
possible_paths = [
    "/kaggle/input/isl-csltr-indian-sign-language-dataset",
    "/kaggle/input/indian-sign-language-isl",
    "/kaggle/input/isl",
    "C:/Users/erick/Downloads/isl-csltr-indian-sign-language-dataset",
    "C:/Users/erick/Downloads/indian-sign-language-isl",
    "./isl-csltr-indian-sign-language-dataset",
    "./indian-sign-language-isl",
    "./isl",
]

dataset_path = None
for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        print(f"✅ Found dataset at: {path}")
        break

if dataset_path is None:
    print("❌ Dataset not found!")
    print("\nPlease specify the dataset location:")
    dataset_path = input("Enter path to ISL dataset: ").strip()
    
    if not os.path.exists(dataset_path):
        print(f"❌ Path does not exist: {dataset_path}")
        exit(1)

# ============================================================================
# 2. FIND ALL VIDEOS
# ============================================================================

print("\n📁 Searching for videos...")

video_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')):
            video_files.append(os.path.join(root, file))

if len(video_files) == 0:
    print("❌ No video files found!")
    exit(1)

print(f"✅ Found {len(video_files)} video files")
print(f"   Sample videos:")
for vf in video_files[:5]:
    print(f"   - {Path(vf).name}")

# ============================================================================
# 3. INITIALIZE MODEL
# ============================================================================

print("\n🔧 Initializing model...")

try:
    pipeline = USLInferencePipelineReal(
        model_path='./usl_models/real_video_model.pth',
        device='cpu'
    )
    print("✅ Model initialized!")
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    exit(1)

# ============================================================================
# 4. TEST ON VIDEOS
# ============================================================================

print("\n" + "=" * 80)
print("TESTING ON VIDEOS")
print("=" * 80)

results = []
num_tests = min(20, len(video_files))  # Test up to 20 videos

for i, video_path in enumerate(video_files[:num_tests], 1):
    video_name = Path(video_path).name
    
    print(f"\n[{i}/{num_tests}] Testing: {video_name}")
    
    try:
        result = pipeline.classify_video(video_path)
        
        print(f"   → Class: {result['class']}")
        print(f"   → Confidence: {result['confidence']:.2%}")
        
        if 'error' in result:
            print(f"   ⚠️  {result['error']}")
        
        results.append({
            'video': video_name,
            'class': result['class'],
            'confidence': result['confidence'],
            'all_probs': result.get('all_probs', {}),
            'frames': result.get('frames_extracted', 0)
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({
            'video': video_name,
            'error': str(e)
        })

# ============================================================================
# 5. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

successful = [r for r in results if 'error' not in r]
failed = [r for r in results if 'error' in r]

print(f"\n✅ Successful: {len(successful)}/{len(results)}")
print(f"❌ Failed: {len(failed)}/{len(results)}")

if successful:
    print(f"\n📊 Classification Distribution:")
    
    class_counts = {}
    for r in successful:
        cls = r['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * count
        print(f"   {cls:12} {count:3} {bar}")
    
    print(f"\n📈 Average Confidence by Class:")
    
    class_confs = {}
    for r in successful:
        cls = r['class']
        if cls not in class_confs:
            class_confs[cls] = []
        class_confs[cls].append(r['confidence'])
    
    for cls in sorted(class_confs.keys()):
        avg_conf = sum(class_confs[cls]) / len(class_confs[cls])
        print(f"   {cls:12} {avg_conf:.2%}")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print(f"\n💾 Saving results...")

with open('./test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to: ./test_results.json")

print("\n" + "=" * 80)
print("✅ TESTING COMPLETE!")
print("=" * 80)
