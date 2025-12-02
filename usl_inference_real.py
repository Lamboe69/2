#!/usr/bin/env python3
"""
USL System - Inference Pipeline with Real-Trained Model
Uses the model trained on actual ISL videos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Dict, Tuple

# ============================================================================
# REAL-TRAINED MODEL ARCHITECTURE
# ============================================================================

class RealTrainedModel(nn.Module):
    """Model trained on real ISL videos"""
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.embedding = nn.Linear(3, 64)
        self.lstm = nn.LSTM(64 * 33, 128, 2, batch_first=True, dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, pose):
        batch_size, seq_len, joints, coords = pose.shape
        
        x = self.embedding(pose)
        x = x.reshape(batch_size, seq_len, -1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        
        logits = self.fc(last_hidden)
        return logits

# ============================================================================
# POSE EXTRACTOR
# ============================================================================

class PoseExtractor:
    """Extract poses from videos using MediaPipe"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_pose(self, video_path, max_frames=150):
        """Extract pose from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        pose_sequence = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_holistic.process(rgb_frame)
            
            if results.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            else:
                pose = np.zeros((33, 3))
            
            pose_sequence.append(pose)
            frame_count += 1
        
        cap.release()
        return np.array(pose_sequence) if pose_sequence else None

# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class USLInferencePipelineReal:
    """Complete inference pipeline with real-trained model"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to real_video_model.pth
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint['config']
        self.class_names = config['class_names']
        self.idx_to_label = config['idx_to_label']
        
        self.model = RealTrainedModel(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Pose extractor
        self.extractor = PoseExtractor()
        
        print(f"✅ USL Inference Pipeline initialized on {self.device}")
        print(f"   Classes: {self.class_names}")
    
    def classify_video(self, video_path: str) -> Dict:
        """Classify a video"""
        
        # Extract pose
        pose = self.extractor.extract_pose(video_path)
        
        if pose is None or len(pose) < 10:
            return {
                'class': 'unknown',
                'confidence': 0.0,
                'error': 'Could not extract pose'
            }
        
        # Pad to 150 frames
        if pose.shape[0] < 150:
            pad_len = 150 - pose.shape[0]
            pose = np.pad(pose, ((0, pad_len), (0, 0), (0, 0)))
        else:
            pose = pose[:150]
        
        # Convert to tensor
        pose_tensor = torch.FloatTensor(pose).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(pose_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        class_name = self.class_names[pred_idx]
        
        return {
            'class': class_name,
            'confidence': confidence,
            'all_probs': {self.class_names[i]: float(probs[0, i].item()) for i in range(len(self.class_names))},
            'frames_extracted': len(pose)
        }
    
    def process_video(self, video_path: str) -> Dict:
        """Complete pipeline: video → pose → classification"""
        print(f"\n📹 Processing video: {video_path}")
        
        result = self.classify_video(video_path)
        
        print(f"   ✅ Class: {result['class']}")
        print(f"   ✅ Confidence: {result['confidence']:.2%}")
        print(f"   ✅ Frames: {result.get('frames_extracted', 'N/A')}")
        
        return result

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = USLInferencePipelineReal(
        model_path='./usl_models/real_video_model.pth',
        device='cuda'
    )
    
    # Example: Process a video
    # result = pipeline.process_video('sample_video.mp4')
    # print(result)
    
    print("\n✅ USL Inference Pipeline ready for use!")
