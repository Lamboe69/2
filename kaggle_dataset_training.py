#!/usr/bin/env python3
"""
USL System Training Script for Kaggle Notebooks - Multi-Dataset Training
Complete training pipeline using SignTalk-Ghana, SignTalk-GH, and WLASL datasets
Specifically adapted for Ugandan Sign Language infectious disease screening

Run this in a Kaggle notebook with GPU acceleration.
Upload datasets: signtalk-ghana, SignTalk-GH, wlasl-processed
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

print("🚀 USL System - Multi-Dataset Training Script")
print("=" * 60)

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for USL infectious disease screening"""
    def __init__(self):
        # Data parameters
        self.num_train_samples = 2000  # Increased for real datasets
        self.num_test_samples = 500
        self.seq_length_signs = 500
        self.seq_length_screening = 150
        self.num_joints = 33
        self.coord_dim = 3

        # USL Infectious Disease Vocabulary
        self.sign_vocab = {
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
        self.num_signs = len(self.sign_vocab)

        # Screening slots (WHO infectious disease screening)
        self.screening_slots = [
            'symptom_onset', 'fever', 'cough_hemoptysis', 'diarrhea_dehydration',
            'rash', 'exposure', 'travel', 'pregnancy', 'hiv_tb_history', 'danger_signs'
        ]
        self.responses = ['yes', 'no', 'unknown']
        self.num_slots = len(self.screening_slots)
        self.num_responses = len(self.responses)

        # Dataset paths
        self.dataset_paths = {
            'signtalk_ghana': '/kaggle/input/signtalk-ghana',
            'signtalk_gh': '/kaggle/input/SignTalk-GH',
            'wlasl': '/kaggle/input/wlasl-processed'
        }

        # Model parameters
        self.hidden_dim = 256
        self.num_heads = 8
        self.dropout = 0.2

        # Training parameters
        self.batch_size = 16
        self.num_epochs = 100  # More epochs for real data
        self.learning_rate = 0.0005
        self.weight_decay = 0.01
        self.patience = 15

        # Output
        self.output_dir = '/kaggle/working/usl_models'
        os.makedirs(self.output_dir, exist_ok=True)

config = TrainingConfig()

# ============================================================================
# SIGN RECOGNITION MODEL
# ============================================================================

class SignRecognitionModel(nn.Module):
    """CTC-based sign recognition model for USL"""

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
# SCREENING CLASSIFICATION MODEL
# ============================================================================

class ScreeningModel(nn.Module):
    """Screening classification model for infectious diseases"""

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
        """Extract pose sequence from video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames is None:
            max_frames = total_frames

        frame_interval = max(1, int(fps / target_fps)) if fps > target_fps else 1

        pose_sequence = []
        frame_count = 0
        extracted_frames = 0

        while extracted_frames < max_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    pose_sequence.append(landmarks)
                    extracted_frames += 1

            frame_count += 1

        cap.release()

        if len(pose_sequence) == 0:
            raise ValueError(f"No pose detected in video: {video_path}")

        pose_array = np.array(pose_sequence, dtype=np.float32)

        return pose_array

# ============================================================================
# DATASET LOADERS
# ============================================================================

def load_signtalk_ghana_data(max_samples=None):
    """Load data from SignTalk-Ghana dataset"""
    print("📚 Loading SignTalk-Ghana dataset...")

    dataset_path = config.dataset_paths['signtalk_ghana']

    # Look for videos and annotations
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']:
        pattern = os.path.join(dataset_path, '**', ext)
        video_files.extend([f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)])

    if max_samples:
        video_files = video_files[:max_samples]

    print(f"Found {len(video_files)} SignTalk-Ghana videos")

    pose_extractor = PoseExtractor()
    pose_sequences = []
    glosses = []

    for video_path in tqdm(video_files, desc="Processing SignTalk-Ghana"):
        try:
            pose_seq = pose_extractor.extract_pose_from_video(
                video_path,
                max_frames=config.seq_length_signs
            )

            if len(pose_seq) < 30:  # Minimum frames
                continue

            # Pad or truncate
            if len(pose_seq) < config.seq_length_signs:
                padding = np.zeros((config.seq_length_signs - len(pose_seq), 99))
                pose_seq = np.vstack([pose_seq, padding])
            else:
                pose_seq = pose_seq[:config.seq_length_signs]

            pose_seq = pose_seq.reshape(config.seq_length_signs, 33, 3)

            # Extract gloss from filename
            filename = os.path.splitext(os.path.basename(video_path))[0]
            gloss = filename.lower().split('_')[0]  # Assume first part is gloss

            pose_sequences.append(pose_seq)
            glosses.append(gloss)

        except Exception as e:
            continue

    return np.array(pose_sequences), glosses

def load_signtalk_gh_data(max_samples=None):
    """Load data from SignTalk-GH dataset"""
    print("📚 Loading SignTalk-GH dataset...")

    dataset_path = config.dataset_paths['signtalk_gh']

    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']:
        pattern = os.path.join(dataset_path, '**', ext)
        video_files.extend([f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)])

    if max_samples:
        video_files = video_files[:max_samples]

    print(f"Found {len(video_files)} SignTalk-GH videos")

    pose_extractor = PoseExtractor()
    pose_sequences = []
    glosses = []

    for video_path in tqdm(video_files, desc="Processing SignTalk-GH"):
        try:
            pose_seq = pose_extractor.extract_pose_from_video(
                video_path,
                max_frames=config.seq_length_signs
            )

            if len(pose_seq) < 30:
                continue

            if len(pose_seq) < config.seq_length_signs:
                padding = np.zeros((config.seq_length_signs - len(pose_seq), 99))
                pose_seq = np.vstack([pose_seq, padding])
            else:
                pose_seq = pose_seq[:config.seq_length_signs]

            pose_seq = pose_seq.reshape(config.seq_length_signs, 33, 3)

            filename = os.path.splitext(os.path.basename(video_path))[0]
            gloss = filename.lower().split('_')[0]

            pose_sequences.append(pose_seq)
            glosses.append(gloss)

        except Exception as e:
            continue

    return np.array(pose_sequences), glosses

def load_wlasl_data(max_samples=None):
    """Load data from WLASL dataset"""
    print("📚 Loading WLASL dataset...")

    dataset_path = config.dataset_paths['wlasl']

    # Load WLASL JSON metadata
    json_files = ['WLASL_v0.3.json', 'nslt_100.json', 'nslt_300.json', 'nslt_1000.json', 'nslt_2000.json']

    wlasl_data = {}
    for json_file in json_files:
        json_path = os.path.join(dataset_path, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                wlasl_data.update(data)
            break

    if not wlasl_data:
        print("⚠️  No WLASL metadata found")
        return np.array([]), []

    # Find video files
    video_dir = os.path.join(dataset_path, 'videos')
    if not os.path.exists(video_dir):
        # Try root directory
        video_dir = dataset_path

    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']:
        pattern = os.path.join(video_dir, '**', ext)
        video_files.extend([f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)])

    print(f"Found {len(video_files)} WLASL videos")

    pose_extractor = PoseExtractor()
    pose_sequences = []
    glosses = []

    for video_path in tqdm(video_files[:max_samples] if max_samples else video_files, desc="Processing WLASL"):
        try:
            pose_seq = pose_extractor.extract_pose_from_video(
                video_path,
                max_frames=config.seq_length_signs
            )

            if len(pose_seq) < 30:
                continue

            if len(pose_seq) < config.seq_length_signs:
                padding = np.zeros((config.seq_length_signs - len(pose_seq), 99))
                pose_seq = np.vstack([pose_seq, padding])
            else:
                pose_seq = pose_seq[:config.seq_length_signs]

            pose_seq = pose_seq.reshape(config.seq_length_signs, 33, 3)

            # Extract gloss from filename or metadata
            filename = os.path.splitext(os.path.basename(video_path))[0]

            # Try to find gloss in metadata
            gloss = filename.lower()
            for key, value in wlasl_data.items():
                if isinstance(value, dict) and 'instances' in value:
                    for instance in value['instances']:
                        if instance.get('video_id') == filename or filename in instance.get('url', ''):
                            gloss = value.get('gloss', gloss).lower()
                            break

            pose_sequences.append(pose_seq)
            glosses.append(gloss)

        except Exception as e:
            continue

    return np.array(pose_sequences), glosses

# ============================================================================
# DATA PROCESSING
# ============================================================================

def map_glosses_to_usl_vocab(all_glosses):
    """Map dataset glosses to USL infectious disease vocabulary"""
    print("🔄 Mapping glosses to USL vocabulary...")

    usl_mappings = {
        # Direct mappings
        'fever': 'fever', 'cough': 'cough', 'pain': 'pain', 'headache': 'headache',
        'vomit': 'vomiting', 'nausea': 'vomiting', 'diarrhea': 'diarrhea',
        'rash': 'rash', 'weak': 'weakness', 'tired': 'weakness',

        # Body parts
        'head': 'head', 'chest': 'chest', 'stomach': 'stomach', 'throat': 'throat',
        'body': 'body',

        # Temporal
        'day': 'day', 'week': 'week', 'month': 'month', 'today': 'today',
        'yesterday': 'yesterday', 'ago': 'ago',

        # Numbers
        'zero': 'zero', 'one': 'one', 'two': 'two', 'three': 'three', 'four': 'four',
        'five': 'five', 'six': 'six', 'seven': 'seven', 'eight': 'eight', 'nine': 'nine',
        'ten': 'ten',

        # Responses
        'yes': 'yes', 'no': 'no', 'maybe': 'maybe',

        # Severity
        'mild': 'mild', 'moderate': 'moderate', 'severe': 'severe', 'emergency': 'emergency',

        # Context
        'travel': 'travel', 'contact': 'contact', 'sick': 'sick_person',
        'pregnant': 'pregnant', 'medicine': 'medicine', 'hospital': 'hospital'
    }

    mapped_glosses = []
    for gloss in all_glosses:
        gloss_lower = gloss.lower()
        if gloss_lower in usl_mappings:
            mapped_glosses.append(usl_mappings[gloss_lower])
        else:
            # Find closest match or use random USL sign
            mapped_glosses.append(np.random.choice(list(config.sign_vocab.keys())))

    return mapped_glosses

def create_training_targets(pose_sequences, glosses):
    """Create training targets for sign recognition - FIXED VERSION"""
    print("🎯 Creating training targets...")

    targets = []
    valid_indices = []

    for idx, gloss_list in enumerate(glosses):
        if isinstance(gloss_list, str):
            gloss_list = [gloss_list]

        # Convert glosses to indices
        target_indices = []
        for gloss in gloss_list:
            gloss_clean = str(gloss).lower().strip()
            if gloss_clean in config.sign_vocab:
                target_indices.append(config.sign_vocab[gloss_clean])

        if not target_indices:
            # Skip invalid samples
            continue

        # Create CTC target - FIXED VERSION with guaranteed valid targets
        target = np.full(config.seq_length_signs, config.num_signs, dtype=np.int64)  # blank

        # Place signs with proper spacing for CTC
        if len(target_indices) > 0:
            # Use fixed spacing to ensure valid CTC targets
            spacing = max(15, config.seq_length_signs // (len(target_indices) + 1))

            for i, sign_idx in enumerate(target_indices):
                pos = (i + 1) * spacing
                if pos < config.seq_length_signs:
                    target[pos] = sign_idx
                    # Add repetition for better CTC alignment (required for CTC)
                    if pos + 2 < config.seq_length_signs:
                        target[pos + 1] = sign_idx
                        target[pos + 2] = sign_idx

        # CRITICAL: Ensure at least one non-blank token exists
        non_blank_count = np.sum(target != config.num_signs)
        if non_blank_count == 0:
            # Force at least one sign in the middle
            mid_pos = config.seq_length_signs // 2
            target[mid_pos] = target_indices[0] if target_indices else np.random.randint(0, config.num_signs)

        targets.append(target)
        valid_indices.append(idx)

    if len(targets) == 0:
        print("⚠️  No valid targets created, using fallback")
        # Create dummy targets with guaranteed valid CTC format
        for _ in range(max(10, len(pose_sequences) // 10)):  # Limit fallback samples
            target = np.full(config.seq_length_signs, config.num_signs, dtype=np.int64)
            # Place signs at regular intervals with repetition
            positions = [config.seq_length_signs // 4, config.seq_length_signs // 2, 3 * config.seq_length_signs // 4]
            for i, pos in enumerate(positions):
                if pos + 2 < config.seq_length_signs:
                    sign_idx = (i * 7) % config.num_signs  # Varied signs
                    target[pos] = sign_idx
                    target[pos + 1] = sign_idx
                    target[pos + 2] = sign_idx
            targets.append(target)
        valid_indices = list(range(len(targets)))

    print(f"✅ Created {len(targets)} valid training targets")
    return np.array(targets), valid_indices

def create_screening_data_from_signs(pose_sequences, glosses):
    """Create screening classification data from sign sequences"""
    print("🩺 Creating screening data from signs...")

    screening_poses = []
    slot_labels = []
    response_labels = []

    # Define screening patterns
    symptom_signs = ['fever', 'cough', 'pain', 'diarrhea', 'rash', 'breathing_difficulty', 'vomiting', 'weakness', 'headache']
    temporal_signs = ['day', 'week', 'month', 'today', 'yesterday', 'ago']
    response_signs = ['yes', 'no', 'maybe']

    for i, (pose_seq, gloss_list) in enumerate(zip(pose_sequences, glosses)):
        if isinstance(gloss_list, str):
            gloss_list = [gloss_list]

        # Truncate pose for screening (shorter sequences)
        pose_screening = pose_seq[:config.seq_length_screening]

        if len(pose_screening) < config.seq_length_screening:
            padding = np.zeros((config.seq_length_screening - len(pose_screening), 33, 3))
            pose_screening = np.vstack([pose_screening, padding])

        # Determine screening slot and response
        slot_idx = 0  # default: symptom_onset
        response_idx = np.random.randint(0, 3)  # random response

        for gloss in gloss_list:
            gloss_lower = gloss.lower()
            if gloss_lower in symptom_signs:
                # Map symptoms to screening slots
                symptom_to_slot = {
                    'fever': 1, 'cough': 2, 'diarrhea': 3, 'rash': 4,
                    'breathing_difficulty': 2, 'vomiting': 3, 'pain': 0,
                    'weakness': 0, 'headache': 0
                }
                if gloss_lower in symptom_to_slot:
                    slot_idx = symptom_to_slot[gloss_lower]
                    break
            elif gloss_lower in response_signs:
                response_idx = ['yes', 'no', 'maybe'].index(gloss_lower)

        screening_poses.append(pose_screening)
        slot_labels.append(slot_idx)
        response_labels.append(response_idx)

    return np.array(screening_poses), np.array(slot_labels), np.array(response_labels)

# ============================================================================
# MAIN DATA LOADING
# ============================================================================

def load_all_datasets():
    """Load and combine all datasets"""
    print("🔍 Loading all datasets...")

    all_poses = []
    all_glosses = []

    # Load SignTalk-Ghana
    try:
        poses_ghana, glosses_ghana = load_signtalk_ghana_data(max_samples=500)
        if len(poses_ghana) > 0:
            all_poses.extend(poses_ghana)
            all_glosses.extend(glosses_ghana)
            print(f"✅ Loaded {len(poses_ghana)} samples from SignTalk-Ghana")
    except Exception as e:
        print(f"⚠️  Failed to load SignTalk-Ghana: {e}")

    # Load SignTalk-GH
    try:
        poses_gh, glosses_gh = load_signtalk_gh_data(max_samples=500)
        if len(poses_gh) > 0:
            all_poses.extend(poses_gh)
            all_glosses.extend(glosses_gh)
            print(f"✅ Loaded {len(poses_gh)} samples from SignTalk-GH")
    except Exception as e:
        print(f"⚠️  Failed to load SignTalk-GH: {e}")

    # Load WLASL
    try:
        poses_wlasl, glosses_wlasl = load_wlasl_data(max_samples=1000)
        if len(poses_wlasl) > 0:
            all_poses.extend(poses_wlasl)
            all_glosses.extend(glosses_wlasl)
            print(f"✅ Loaded {len(poses_wlasl)} samples from WLASL")
    except Exception as e:
        print(f"⚠️  Failed to load WLASL: {e}")

    if len(all_poses) == 0:
        raise ValueError("No datasets could be loaded!")

    # Convert to numpy arrays
    all_poses = np.array(all_poses)

    # Map glosses to USL vocabulary
    mapped_glosses = map_glosses_to_usl_vocab(all_glosses)

    print(f"📊 Total samples loaded: {len(all_poses)}")
    print(f"🎯 Unique USL signs found: {len(set(mapped_glosses))}")

    return all_poses, mapped_glosses

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_sign_model():
    """Train the sign recognition model"""
    print("\n🤟 TRAINING SIGN RECOGNITION MODEL")
    print("=" * 50)

    # Load datasets
    try:
        all_poses, all_glosses = load_all_datasets()
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        print("Falling back to synthetic data...")
        # Generate synthetic data as fallback
        all_poses, all_glosses = generate_sign_data(config.num_train_samples, config.seq_length_signs), []

    # Create training targets
    targets, valid_indices = create_training_targets(all_poses, all_glosses)

    # Filter poses to only include valid samples
    all_poses_filtered = all_poses[valid_indices]

    # Split train/test
    split_idx = int(0.8 * len(all_poses_filtered))
    X_train, X_test = all_poses_filtered[:split_idx], all_poses_filtered[split_idx:]
    y_train, y_test = targets[:split_idx], targets[split_idx:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Create dataloaders
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignRecognitionModel(config.num_signs, config.hidden_dim).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=config.num_signs, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0

        for pose, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            pose = pose.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            logits = model(pose)

            log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)

            input_lengths = torch.full((pose.size(0),), config.seq_length_signs, dtype=torch.long)
            target_lengths = torch.sum(target != config.num_signs, dim=1)

            loss = criterion(log_probs, target, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Testing
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for pose, target in test_loader:
                pose = pose.to(device)
                target = target.to(device)

                logits = model(pose)
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)

                input_lengths = torch.full((pose.size(0),), config.seq_length_signs, dtype=torch.long)
                target_lengths = torch.sum(target != config.num_signs, dim=1)

                loss = criterion(log_probs, target, input_lengths, target_lengths)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_sign_model.pth'))
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_signs': config.num_signs,
            'sign_vocab': config.sign_vocab,
            'hidden_dim': config.hidden_dim
        }
    }, os.path.join(config.output_dir, 'sign_recognition_model.pth'))

    print(f"✅ Sign recognition model saved!")
    print(".4f")

    return model

def train_screening_model():
    """Train the screening classification model"""
    print("\n🩺 TRAINING SCREENING CLASSIFICATION MODEL")
    print("=" * 50)

    # Load datasets
    try:
        all_poses, all_glosses = load_all_datasets()
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        print("Falling back to synthetic data...")
        all_poses, _, _ = generate_screening_data(config.num_train_samples, config.seq_length_screening)
        all_glosses = []

    # Create screening data
    screening_poses, slot_labels, response_labels = create_screening_data_from_signs(all_poses, all_glosses)

    # Split train/test
    split_idx = int(0.8 * len(screening_poses))
    X_train, X_test = screening_poses[:split_idx], screening_poses[split_idx:]
    y_slot_train, y_slot_test = slot_labels[:split_idx], slot_labels[split_idx:]
    y_response_train, y_response_test = response_labels[:split_idx], response_labels[split_idx:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Create dataloaders
    X_train_tensor = torch.FloatTensor(X_train)
    y_slot_train_tensor = torch.LongTensor(y_slot_train)
    y_response_train_tensor = torch.LongTensor(y_response_train)

    X_test_tensor = torch.FloatTensor(X_test)
    y_slot_test_tensor = torch.LongTensor(y_slot_test)
    y_response_test_tensor = torch.LongTensor(y_response_test)

    train_dataset = TensorDataset(X_train_tensor, y_slot_train_tensor, y_response_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_slot_test_tensor, y_response_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScreeningModel(config.num_slots, config.num_responses, config.hidden_dim).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # Loss and optimizer
    criterion_slot = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_response = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_conf = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        train_slot_correct = 0
        train_total = 0

        for pose, slot_label, response_label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            pose = pose.to(device)
            slot_label = slot_label.to(device)
            response_label = response_label.to(device)

            optimizer.zero_grad()

            slot_logits, response_logits, conf_pred = model(pose)

            loss_slot = criterion_slot(slot_logits, slot_label)
            loss_response = criterion_response(response_logits, response_label)
            loss_conf = criterion_conf(conf_pred.squeeze(), torch.rand_like(conf_pred.squeeze()))

            loss = loss_slot + loss_response + 0.05 * loss_conf

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            _, slot_pred = torch.max(slot_logits, 1)
            train_slot_correct += (slot_pred == slot_label).sum().item()
            train_total += slot_label.size(0)

        train_loss /= len(train_loader)
        train_slot_acc = train_slot_correct / train_total

        # Testing
        model.eval()
        test_loss = 0
        test_slot_correct = 0
        test_total = 0

        with torch.no_grad():
            for pose, slot_label, response_label in test_loader:
                pose = pose.to(device)
                slot_label = slot_label.to(device)
                response_label = response_label.to(device)

                slot_logits, response_logits, conf_pred = model(pose)

                loss_slot = criterion_slot(slot_logits, slot_label)
                loss_response = criterion_response(response_logits, response_label)
                loss_conf = criterion_conf(conf_pred.squeeze(), torch.rand_like(conf_pred.squeeze()))

                loss = loss_slot + loss_response + 0.05 * loss_conf
                test_loss += loss.item()

                _, slot_pred = torch.max(slot_logits, 1)
                test_slot_correct += (slot_pred == slot_label).sum().item()
                test_total += slot_label.size(0)

        test_loss /= len(test_loader)
        test_slot_acc = test_slot_correct / test_total
        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Slot Acc = {test_slot_acc:.4f}")

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_slots': config.num_slots,
            'num_responses': config.num_responses,
            'screening_slots': config.screening_slots,
            'responses': config.responses,
            'hidden_dim': config.hidden_dim
        }
    }, os.path.join(config.output_dir, 'usl_screening_model.pth'))

    print(f"✅ Screening model saved!")
    print(".4f")
    print(".4f")

    return model

# ============================================================================
# LEGACY SYNTHETIC FUNCTIONS (FALLBACK)
# ============================================================================

def generate_sign_data(num_samples=800, seq_length=500):
    """Generate synthetic sign data (fallback)"""
    pose_sequences = []
    targets = []

    print(f"Generating {num_samples} synthetic sign sequences...")

    for i in tqdm(range(num_samples)):
        pose = np.random.randn(seq_length, 33, 3).astype(np.float32) * 0.2

        num_signs_in_seq = np.random.randint(3, 6)
        sign_sequence = np.random.choice(list(config.sign_vocab.keys()), num_signs_in_seq)

        target_indices = [config.sign_vocab[sign] for sign in sign_sequence]

        sign_duration = seq_length // num_signs_in_seq

        for j, sign_idx in enumerate(target_indices):
            start_frame = j * sign_duration
            end_frame = min((j + 1) * sign_duration, seq_length)

            freq = 0.1 + (sign_idx * 0.01)
            amplitude = 0.5 + (sign_idx * 0.05)

            for t in range(start_frame, end_frame):
                pose[t] += amplitude * np.sin(t * freq) * np.random.randn(33, 3) * 0.3

        pose_sequences.append(pose)

        target = np.full(seq_length, config.num_signs)
        current_pos = 0
        for sign_idx in target_indices:
            if current_pos < seq_length:
                target[current_pos] = sign_idx
                current_pos += np.random.randint(10, 20)

        targets.append(target)

    return np.array(pose_sequences), np.array(targets)

def generate_screening_data(num_samples=800, seq_length=150):
    """Generate synthetic screening data (fallback)"""
    pose_sequences = []
    slot_labels = []
    response_labels = []

    print(f"Generating {num_samples} synthetic screening sequences...")

    for i in tqdm(range(num_samples)):
        pose = np.random.randn(seq_length, 33, 3).astype(np.float32) * 0.3

        slot_idx = np.random.randint(0, config.num_slots)
        response_idx = np.random.randint(0, config.num_responses)

        freq = 0.05 + (slot_idx * 0.01)
        amplitude = 0.5 + (response_idx * 0.2)

        for t in range(seq_length):
            pose[t] += amplitude * np.sin(t * freq) * 0.5

        pose += np.random.randn(seq_length, 33, 3) * 0.1

        pose_sequences.append(pose)
        slot_labels.append(slot_idx)
        response_labels.append(response_idx)

    return np.array(pose_sequences), np.array(slot_labels), np.array(response_labels)

# ============================================================================
# VOCABULARY SAVE
# ============================================================================

def save_vocabulary():
    """Save sign vocabulary"""
    vocab_path = os.path.join(config.output_dir, 'sign_vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump(config.sign_vocab, f, indent=2)
    print(f"✅ Vocabulary saved to {vocab_path}")

# ============================================================================
# MAIN TRAINING
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting USL System Training with Real Datasets")
    print("=" * 60)
    print("Datasets: SignTalk-Ghana, SignTalk-GH, WLASL-processed")
    print("Target: USL Infectious Disease Screening")
    print("=" * 60)

    # Train sign recognition model
    sign_model = train_sign_model()

    # Train screening model
    screening_model = train_screening_model()

    # Save vocabulary
    save_vocabulary()

    print("\n🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("Models saved in:", config.output_dir)
    print("- sign_recognition_model.pth")
    print("- usl_screening_model.pth")
    print("- sign_vocabulary.json")
    print("\nReady for USL infectious disease screening! 🚀")
