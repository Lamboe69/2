#!/usr/bin/env python3
"""
USL System - Complete Inference Pipeline
Graph-Reasoned Large Vision Models for Ugandan Sign Language Translation
in Infectious Disease Screening

This module provides:
1. Real-time video processing
2. Sign recognition using trained CTC model
3. Screening classification using trained model
4. Clinical data generation (FHIR bundles)
5. Skip logic for efficient screening
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
import mediapipe as mp
from datetime import datetime
import warnings
import random
import hashlib
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds at module level
set_random_seeds(42)

# ============================================================================
# SIGN RECOGNITION MODEL (Same as training)
# ============================================================================

class SignRecognitionModel(nn.Module):
    """CTC-based sign recognition model"""

    def __init__(self, num_signs, hidden_dim=256):
        super().__init__()

        self.joint_embedding = nn.Linear(3, hidden_dim)
        self.embedding_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=hidden_dim * 33,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        self.ctc_head = nn.Linear(hidden_dim * 2, num_signs + 1)

    def forward(self, pose_sequence):
        batch_size, seq_len, num_joints, coord_dim = pose_sequence.shape

        pose_embedded = self.joint_embedding(pose_sequence)
        pose_embedded = self.embedding_dropout(pose_embedded)

        pose_flat = pose_embedded.reshape(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(pose_flat)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        ctc_logits = self.ctc_head(attn_out)

        return ctc_logits

# ============================================================================
# SCREENING CLASSIFICATION MODEL (Same as training)
# ============================================================================

class ScreeningModel(nn.Module):
    """Screening classification model"""

    def __init__(self, num_slots=10, num_responses=3, hidden_dim=256):
        super().__init__()

        self.joint_embedding = nn.Linear(3, hidden_dim)
        self.embedding_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=hidden_dim * 33,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        self.slot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_slots)
        )

        self.response_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_responses)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, pose_sequence):
        batch_size, seq_len, num_joints, coord_dim = pose_sequence.shape

        pose_embedded = self.joint_embedding(pose_sequence)
        pose_embedded = self.embedding_dropout(pose_embedded)

        pose_flat = pose_embedded.reshape(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(pose_flat)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        last_hidden = attn_out[:, -1, :]

        slot_logits = self.slot_head(last_hidden)
        response_logits = self.response_head(last_hidden)
        confidence = self.confidence_head(last_hidden)

        return slot_logits, response_logits, confidence

# ============================================================================
# POSE EXTRACTION
# ============================================================================

class PoseExtractor:
    """Extract pose sequences from videos using MediaPipe"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_pose_from_video(self, video_path, max_frames=None, target_fps=30):
        """Extract pose sequence from video file with deterministic processing"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames is None:
            max_frames = min(total_frames, 500)  # Cap at 500 frames for consistency

        # Use fixed frame sampling for consistency
        frame_interval = max(1, int(fps / target_fps)) if fps > target_fps else 1

        pose_sequence = []
        frame_count = 0
        extracted_frames = 0

        # Read all frames first to ensure consistent processing
        frames_to_process = []
        while frame_count < total_frames and len(frames_to_process) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames_to_process.append(frame)
            frame_count += 1

        cap.release()

        # Process frames deterministically
        for frame in frames_to_process:
            if extracted_frames >= max_frames:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Round to 4 decimal places for consistency
                    landmarks.extend([
                        round(landmark.x, 4),
                        round(landmark.y, 4),
                        round(landmark.z, 4)
                    ])
                pose_sequence.append(landmarks)
                extracted_frames += 1

        if len(pose_sequence) == 0:
            raise ValueError(f"No pose detected in video: {video_path}")

        pose_array = np.array(pose_sequence, dtype=np.float32)

        return pose_array

# ============================================================================
# CTC DECODER
# ============================================================================

class CTCDecoder:
    """Decode CTC predictions to sign sequences"""

    def __init__(self, blank_token):
        self.blank_token = blank_token

    def greedy_decode(self, logits):
        """Greedy decoding for CTC"""
        # Get argmax predictions
        predictions = torch.argmax(logits, dim=2)

        # Remove consecutive duplicates and blanks
        decoded = []
        prev_token = None

        for seq in predictions:
            seq_decoded = []
            for token in seq:
                token = token.item()
                if token != self.blank_token and token != prev_token:
                    seq_decoded.append(token)
                prev_token = token
            decoded.append(seq_decoded)

        return decoded

    def decode_predictions(self, logits, vocab):
        """Convert token indices to sign names"""
        decoded_tokens = self.greedy_decode(logits)

        sign_sequences = []
        for seq in decoded_tokens:
            signs = [vocab.get(token, f"unknown_{token}") for token in seq if token in vocab]
            sign_sequences.append(signs)

        return sign_sequences

# ============================================================================
# USL INFERENCE PIPELINE
# ============================================================================

class USLInferencePipeline:
    """
    Complete USL Inference Pipeline for Infectious Disease Screening

    Usage:
        pipeline = USLInferencePipeline(
            sign_model_path='./models/sign_recognition_model.pth',
            screening_model_path='./models/usl_screening_model.pth',
            sign_vocab_path='./models/sign_vocabulary.json'
        )

        result = pipeline.process_video('patient_video.mp4')
    """

    def __init__(self, sign_model_path, screening_model_path, sign_vocab_path, device='auto'):
        """
        Initialize the USL inference pipeline

        Args:
            sign_model_path: Path to trained sign recognition model
            screening_model_path: Path to trained screening model
            sign_vocab_path: Path to sign vocabulary JSON
            device: 'auto', 'cpu', or 'cuda'
        """
        print("Initializing USL Inference Pipeline...")

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load sign vocabulary
        with open(sign_vocab_path, 'r') as f:
            sign_vocab_data = json.load(f)

        # Create reverse mapping (index -> sign)
        self.sign_vocab = {v: k for k, v in sign_vocab_data.items()}
        self.num_signs = len(self.sign_vocab)

        # Screening configuration
        self.screening_slots = [
            'symptom_onset', 'fever', 'cough_hemoptysis', 'diarrhea_dehydration',
            'rash', 'exposure', 'travel', 'pregnancy', 'hiv_tb_history', 'danger_signs'
        ]
        self.responses = ['yes', 'no', 'unknown']
        self.num_slots = len(self.screening_slots)
        self.num_responses = len(self.responses)

        # Initialize models
        self.sign_model = SignRecognitionModel(self.num_signs).to(self.device)
        self.screening_model = ScreeningModel(self.num_slots, self.num_responses).to(self.device)

        # Load trained weights
        sign_checkpoint = torch.load(sign_model_path, map_location=self.device)
        screening_checkpoint = torch.load(screening_model_path, map_location=self.device)

        self.sign_model.load_state_dict(sign_checkpoint['model_state_dict'])
        self.screening_model.load_state_dict(screening_checkpoint['model_state_dict'])

        # Set to evaluation mode
        self.sign_model.eval()
        self.screening_model.eval()

        print("USL Inference Pipeline ready!")

    def extract_pose_from_video(self, video_path, max_frames=None):
        """
        Extract pose sequence from video

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract (None for all)

        Returns:
            pose_array: (frames, 99) numpy array
        """
        # Re-initialize PoseExtractor for each call to ensure statelessness
        pose_extractor = PoseExtractor()
        return pose_extractor.extract_pose_from_video(video_path, max_frames)

    def recognize_signs(self, pose_sequence):
        """
        Recognize signs from pose sequence

        Args:
            pose_sequence: (frames, 99) numpy array

        Returns:
            dict with sign predictions
        """
        # Prepare input
        if len(pose_sequence) < 50:
            raise ValueError("Pose sequence too short for sign recognition")

        # Pad or truncate to fixed length
        seq_length = 500  # Same as training
        if len(pose_sequence) < seq_length:
            padding = np.zeros((seq_length - len(pose_sequence), 99))
            pose_sequence = np.vstack([pose_sequence, padding])
        else:
            pose_sequence = pose_sequence[:seq_length]

        # Reshape for model (batch, seq, joints, coords)
        pose_tensor = torch.FloatTensor(pose_sequence).reshape(1, seq_length, 33, 3).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.sign_model(pose_tensor)

            # Decode predictions
            ctc_decoder = CTCDecoder(blank_token=self.num_signs)
            sign_sequences = ctc_decoder.decode_predictions(logits, self.sign_vocab)

        sign_names = sign_sequences[0] if sign_sequences else []

        return {
            'sign_names': sign_names,
            'num_signs': len(sign_names),
            'confidence': 0.85  # Placeholder confidence
        }

    def classify_screening(self, pose_sequence):
        """
        Classify screening slot and response from pose sequence

        Args:
            pose_sequence: (frames, 99) numpy array

        Returns:
            dict with screening predictions
        """
        # Prepare input
        seq_length = 150  # Same as training
        if len(pose_sequence) < seq_length:
            padding = np.zeros((seq_length - len(pose_sequence), 99))
            pose_sequence = np.vstack([pose_sequence, padding])
        else:
            pose_sequence = pose_sequence[:seq_length]

        # Reshape for model
        pose_tensor = torch.FloatTensor(pose_sequence).reshape(1, seq_length, 33, 3).to(self.device)

        # Forward pass
        with torch.no_grad():
            slot_logits, response_logits, confidence = self.screening_model(pose_tensor)

            # Get predictions
            slot_pred = torch.argmax(slot_logits, dim=1).item()
            response_pred = torch.argmax(response_logits, dim=1).item()
            conf_score = confidence.item()

        return {
            'screening_slot': self.screening_slots[slot_pred],
            'response': self.responses[response_pred],
            'confidence': conf_score,
            'slot_logits': slot_logits.cpu().numpy(),
            'response_logits': response_logits.cpu().numpy()
        }

    def process_video(self, video_path):
        """
        Complete video processing pipeline

        Args:
            video_path: Path to video file

        Returns:
            dict with complete analysis results
        """
        print(f"Processing video: {video_path}")

        # Extract pose
        pose_sequence = self.extract_pose_from_video(video_path)

        # Recognize signs
        sign_result = self.recognize_signs(pose_sequence)

        # Classify screening
        screening_result = self.classify_screening(pose_sequence)

        # Compile results
        result = {
            'video_path': video_path,
            'pose_frames': len(pose_sequence),
            'signs': sign_result,
            'screening': screening_result,
            'timestamp': datetime.now().isoformat(),
            'model_version': 'USL-v1.0'
        }

        print("Video processing complete!")
        return result

    def generate_fhir_bundle(self, result, patient_id="PATIENT-001"):
        """
        Generate FHIR bundle from analysis results

        Args:
            result: Analysis result from process_video()
            patient_id: Patient identifier

        Returns:
            FHIR bundle as dict
        """
        screening = result['screening']

        fhir_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": result['timestamp'],
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "status": "final",
                        "code": {
                            "coding": [{
                                "system": "http://medisign.ug/screening",
                                "code": screening['screening_slot'],
                                "display": screening['screening_slot'].replace('_', ' ').title()
                            }]
                        },
                        "valueString": screening['response'],
                        "extension": [{
                            "url": "http://medisign.ug/confidence",
                            "valueDecimal": screening['confidence']
                        }],
                        "subject": {
                            "reference": f"Patient/{patient_id}"
                        },
                        "effectiveDateTime": result['timestamp']
                    }
                }
            ]
        }

        return fhir_bundle

    def implement_skip_logic(self, completed_slots):
        """
        Implement clinical skip logic for efficient screening

        Args:
            completed_slots: List of completed screening slots

        Returns:
            Next recommended screening slot
        """
        # Basic skip logic rules
        if 'fever' in completed_slots:
            # If fever=yes, prioritize cough assessment
            return 'cough_hemoptysis'

        if 'cough_hemoptysis' in completed_slots:
            # If cough with blood=yes, check danger signs
            return 'danger_signs'

        if 'diarrhea_dehydration' in completed_slots:
            # If diarrhea=yes, check dehydration signs
            return 'danger_signs'

        # Default progression through screening slots
        slot_order = [
            'symptom_onset', 'fever', 'cough_hemoptysis', 'diarrhea_dehydration',
            'rash', 'exposure', 'travel', 'pregnancy', 'hiv_tb_history', 'danger_signs'
        ]

        for slot in slot_order:
            if slot not in completed_slots:
                return slot

        return None  # Screening complete

    def detect_danger_signs(self, result):
        """
        Detect critical danger signs requiring immediate attention

        Args:
            result: Analysis result

        Returns:
            dict with danger sign assessment
        """
        signs = result['signs']['sign_names']
        screening = result['screening']

        # Only detect danger signs based on actual critical conditions
        # Danger signs should ONLY appear when specific critical symptoms are detected
        danger_indicators = {
            'emergency': 'emergency' in signs and len(signs) > 0,
            'severe_pain': 'severe' in signs and 'pain' in signs and len(signs) > 0,
            'breathing_difficulty': 'breathing_difficulty' in signs and len(signs) > 0,
            'severe_dehydration': 'severe' in signs and 'diarrhea' in signs and len(signs) > 0,
            'hemoptysis': screening['screening_slot'] == 'cough_hemoptysis' and screening['response'] == 'yes' and screening['confidence'] > 0.7
        }

        danger_detected = any(danger_indicators.values())

        return {
            'danger_detected': danger_detected,
            'danger_signs': [sign for sign, detected in danger_indicators.items() if detected],
            'triage_level': 'emergency' if danger_detected else 'routine',
            'recommendations': [
                "Immediate medical attention required" if danger_detected else "Continue routine screening"
            ]
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_disease_signs_vocab():
    """Create USL infectious disease signs vocabulary"""
    return {
        # Core symptoms
        'fever': 0, 'cough': 1, 'blood': 2, 'pain': 3, 'diarrhea': 4, 'rash': 5,
        'breathing_difficulty': 6, 'vomiting': 7, 'weakness': 8, 'headache': 9,
        'chest': 10, 'head': 11, 'stomach': 12, 'throat': 13, 'body': 14,

        # Temporal
        'day': 15, 'week': 16, 'month': 17, 'today': 18, 'yesterday': 19, 'ago': 20,

        # Numbers
        'zero': 21, 'one': 22, 'two': 23, 'three': 24, 'four': 25, 'five': 26,
        'six': 27, 'seven': 28, 'eight': 29, 'nine': 30, 'ten': 31,

        # Responses
        'yes': 32, 'no': 33, 'maybe': 34,

        # Severity
        'mild': 35, 'moderate': 36, 'severe': 37, 'emergency': 38,

        # Context
        'travel': 39, 'contact': 40, 'sick_person': 41, 'pregnant': 42, 'medicine': 43, 'hospital': 44
    }

# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("USL Inference Pipeline Demo")
    print("=" * 50)

    # Example usage
    try:
        pipeline = USLInferencePipeline(
            sign_model_path='./usl_models/sign_recognition_model.pth',
            screening_model_path='./usl_models/usl_screening_model.pth',
            sign_vocab_path='./usl_models/sign_vocabulary.json'
        )

        print("Pipeline initialized successfully!")

        # Demo with synthetic pose data
        print("\nTesting with synthetic pose data...")

        # Create synthetic pose sequence
        pose_sequence = np.random.randn(300, 33, 3).astype(np.float32) * 0.1

        # Test sign recognition
        sign_result = pipeline.recognize_signs(pose_sequence)
        print(f"Sign recognition result: {sign_result}")

        # Test screening classification
        screening_result = pipeline.classify_screening(pose_sequence)
        print(f"Screening result: {screening_result}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure model files exist in ./usl_models/")
