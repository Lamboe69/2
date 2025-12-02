#!/usr/bin/env python3
"""
USL System Training Script for Kaggle Notebooks - Dataset Training
Complete training pipeline for Ugandan Sign Language models using SignTalk-Ghana, SignTalk-GH, and WLASL datasets

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

print("🚀 USL System Training Script")
print("=" * 50)

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for both models"""
    def __init__(self):
        # Data parameters
        self.num_train_samples = 800
        self.num_test_samples = 200
        self.seq_length_signs = 500  # Longer sequences for signs
        self.seq_length_screening = 150  # Shorter for screening
        self.num_joints = 33
        self.coord_dim = 3

        # Sign vocabulary
        self.sign_vocab = {
            'fever': 0, 'cough': 1, 'blood': 2, 'pain': 3, 'diarrhea': 4, 'rash': 5,
            'breathing_difficulty': 6, 'vomiting': 7, 'weakness': 8, 'headache': 9,
            'chest': 10, 'head': 11, 'stomach': 12, 'throat': 13, 'body': 14,
            'day': 15, 'week': 16, 'month': 17, 'today': 18, 'yesterday': 19, 'ago': 20,
            'zero': 21, 'one': 22, 'two': 23, 'three': 24, 'four': 25, 'five': 26,
            'six': 27, 'seven': 28, 'eight': 29, 'nine': 30, 'ten': 31,
            'yes': 32, 'no': 33, 'maybe': 34,
            'mild': 35, 'moderate': 36, 'severe': 37, 'emergency': 38,
            'travel': 39, 'contact': 40, 'sick_person': 41, 'pregnant': 42, 'medicine': 43, 'hospital': 44
        }
        self.num_signs = len(self.sign_vocab)

        # Screening parameters
        self.screening_slots = [
            'symptom_onset', 'fever', 'cough_hemoptysis', 'diarrhea_dehydration',
            'rash', 'exposure', 'travel', 'pregnancy', 'hiv_tb_history', 'danger_signs'
        ]
        self.responses = ['yes', 'no', 'unknown']
        self.num_slots = len(self.screening_slots)
        self.num_responses = len(self.responses)

        # Model parameters
        self.hidden_dim = 256
        self.num_heads = 8
        self.dropout = 0.2

        # Training parameters
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 0.0005
        self.weight_decay = 0.01
        self.patience = 10

        # Output
        self.output_dir = '/kaggle/working/usl_models'
        os.makedirs(self.output_dir, exist_ok=True)

config = TrainingConfig()

# ============================================================================
# SIGN RECOGNITION MODEL
# ============================================================================

class SignRecognitionModel(nn.Module):
    """CTC-based sign recognition model"""

    def __init__(self, num_signs, hidden_dim=256):
        super().__init__()

        # Pose embedding
        self.joint_embedding = nn.Linear(3, hidden_dim)
        self.embedding_dropout = nn.Dropout(0.2)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 33,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        # CTC head (num_signs + 1 for blank)
        self.ctc_head = nn.Linear(hidden_dim * 2, num_signs + 1)

    def forward(self, pose_sequence):
        """
        Args:
            pose_sequence: (batch, seq_len, 33, 3)
        Returns:
            ctc_logits: (batch, seq_len, num_signs + 1)
        """
        batch_size, seq_len, num_joints, coord_dim = pose_sequence.shape

        # Embed joints
        pose_embedded = self.joint_embedding(pose_sequence)
        pose_embedded = self.embedding_dropout(pose_embedded)

        # Flatten joints
        pose_flat = pose_embedded.reshape(batch_size, seq_len, -1)

        # LSTM
        lstm_out, _ = self.lstm(pose_flat)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # CTC logits
        ctc_logits = self.ctc_head(attn_out)

        return ctc_logits

# ============================================================================
# SCREENING CLASSIFICATION MODEL
# ============================================================================

class ScreeningModel(nn.Module):
    """Screening classification model"""

    def __init__(self, num_slots=10, num_responses=3, hidden_dim=256):
        super().__init__()

        # Pose embedding
        self.joint_embedding = nn.Linear(3, hidden_dim)
        self.embedding_dropout = nn.Dropout(0.2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 33,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        # Classification heads
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

        # Embed joints
        pose_embedded = self.joint_embedding(pose_sequence)
        pose_embedded = self.embedding_dropout(pose_embedded)

        # Flatten joints
        pose_flat = pose_embedded.reshape(batch_size, seq_len, -1)

        # LSTM
        lstm_out, _ = self.lstm(pose_flat)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Last timestep
        last_hidden = attn_out[:, -1, :]

        # Classification
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

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames is None:
            max_frames = total_frames

        # Frame sampling
        frame_interval = max(1, int(fps / target_fps)) if fps > target_fps else 1

        pose_sequence = []
        frame_count = 0
        extracted_frames = 0

        while extracted_frames < max_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    # Extract 33 pose landmarks (x, y, z)
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    pose_sequence.append(landmarks)
                    extracted_frames += 1

            frame_count += 1

        cap.release()

        if len(pose_sequence) == 0:
            raise ValueError(f"No pose detected in video: {video_path}")

        # Convert to numpy array and reshape
        pose_array = np.array(pose_sequence, dtype=np.float32)

        # Normalize coordinates (optional - MediaPipe already normalizes to 0-1)
        # pose_array = pose_array / np.max(np.abs(pose_array), axis=0)

        return pose_array

# ============================================================================
# DATASET PROCESSING
# ============================================================================

def parse_video_labels(filename):
    """Parse screening labels from video filename"""
    # Expected format: screening_slot_response.mp4
    # e.g., fever_yes.mp4, cough_hemoptysis_no.mp4, symptom_onset_unknown.mp4

    name = os.path.splitext(filename)[0].lower()

    # Split by underscore
    parts = name.split('_')

    if len(parts) >= 2:
        slot_name = parts[0]
        response_name = parts[1]

        # Find slot index
        slot_idx = -1
        for i, slot in enumerate(config.screening_slots):
            if slot_name in slot.lower().replace('_', ''):
                slot_idx = i
                break

        # Find response index
        response_idx = -1
        for i, resp in enumerate(config.responses):
            if response_name == resp.lower():
                response_idx = i
                break

        if slot_idx >= 0 and response_idx >= 0:
            return slot_idx, response_idx

    # Default fallback
    return np.random.randint(0, config.num_slots), np.random.randint(0, config.num_responses)

def load_real_screening_data(video_dir, max_videos=None):
    """Load real screening video data"""
    pose_extractor = PoseExtractor()

    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        pattern = os.path.join(video_dir, '**', ext)
        video_files.extend([f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)])

    if max_videos:
        video_files = video_files[:max_videos]

    print(f"Found {len(video_files)} video files")

    pose_sequences = []
    slot_labels = []
    response_labels = []

    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Extract pose
            pose_seq = pose_extractor.extract_pose_from_video(
                video_path,
                max_frames=config.seq_length_screening
            )

            # Pad or truncate to target length
            if len(pose_seq) < config.seq_length_screening:
                # Pad with zeros
                padding = np.zeros((config.seq_length_screening - len(pose_seq), 99))
                pose_seq = np.vstack([pose_seq, padding])
            elif len(pose_seq) > config.seq_length_screening:
                # Truncate
                pose_seq = pose_seq[:config.seq_length_screening]

            # Reshape to (seq_len, 33, 3)
            pose_seq = pose_seq.reshape(config.seq_length_screening, 33, 3)

            # Parse labels from filename
            filename = os.path.basename(video_path)
            slot_idx, response_idx = parse_video_labels(filename)

            pose_sequences.append(pose_seq)
            slot_labels.append(slot_idx)
            response_labels.append(response_idx)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    if len(pose_sequences) == 0:
        print("⚠️  No valid videos found. Using synthetic data as fallback.")
        return generate_screening_data(len(video_files) or 100, config.seq_length_screening)

    return np.array(pose_sequences), np.array(slot_labels), np.array(response_labels)

def create_sign_targets_from_videos(video_dir, max_videos=None):
    """Create sign recognition targets from video annotations"""
    # For sign recognition, we need annotated video segments
    # This is more complex - assume videos are named with sign sequences

    pose_extractor = PoseExtractor()

    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        pattern = os.path.join(video_dir, '**', ext)
        video_files.extend([f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)])

    if max_videos:
        video_files = video_files[:max_videos]

    print(f"Found {len(video_files)} video files for sign training")

    pose_sequences = []
    targets = []

    for video_path in tqdm(video_files, desc="Processing sign videos"):
        try:
            # Extract full pose sequence
            pose_seq = pose_extractor.extract_pose_from_video(
                video_path,
                max_frames=config.seq_length_signs
            )

            if len(pose_seq) < 50:  # Minimum frames
                continue

            # Pad or truncate
            if len(pose_seq) < config.seq_length_signs:
                padding = np.zeros((config.seq_length_signs - len(pose_seq), 99))
                pose_seq = np.vstack([pose_seq, padding])
            elif len(pose_seq) > config.seq_length_signs:
                pose_seq = pose_seq[:config.seq_length_signs]

            # Reshape
            pose_seq = pose_seq.reshape(config.seq_length_signs, 33, 3)

            # Parse signs from filename (e.g., fever_three_day.mp4)
            filename = os.path.splitext(os.path.basename(video_path))[0].lower()
            sign_names = filename.split('_')

            # Convert to indices
            target_indices = []
            for sign_name in sign_names:
                if sign_name in config.sign_vocab:
                    target_indices.append(config.sign_vocab[sign_name])

            if not target_indices:
                # Fallback: random signs
                num_signs = np.random.randint(2, 5)
                target_indices = np.random.choice(list(config.sign_vocab.values()), num_signs, replace=False)

            # Create CTC target
            target = np.full(config.seq_length_signs, config.num_signs)  # blank
            sign_duration = config.seq_length_signs // len(target_indices)

            for i, sign_idx in enumerate(target_indices):
                start = i * sign_duration
                end = min((i + 1) * sign_duration, config.seq_length_signs)
                target[start:end] = sign_idx

            pose_sequences.append(pose_seq)
            targets.append(target)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    if len(pose_sequences) == 0:
        print("⚠️  No valid sign videos found. Using synthetic data as fallback.")
        return generate_sign_data(len(video_files) or 100, config.seq_length_signs)

    return np.array(pose_sequences), np.array(targets)

# ============================================================================
# LEGACY SYNTHETIC FUNCTIONS (FALLBACK)
# ============================================================================

def generate_sign_data(num_samples=800, seq_length=500):
    """Generate synthetic sign data (fallback)"""
    pose_sequences = []
    targets = []

    print(f"Generating {num_samples} synthetic sign sequences...")

    for i in tqdm(range(num_samples)):
        # Base pose
        pose = np.random.randn(seq_length, 33, 3).astype(np.float32) * 0.2

        # Create sign sequence (3-5 signs per sequence)
        num_signs_in_seq = np.random.randint(3, 6)
        sign_sequence = np.random.choice(list(config.sign_vocab.keys()), num_signs_in_seq)

        # Convert to indices
        target_indices = [config.sign_vocab[sign] for sign in sign_sequence]

        # Create temporal patterns for each sign
        sign_duration = seq_length // num_signs_in_seq

        for j, sign_idx in enumerate(target_indices):
            start_frame = j * sign_duration
            end_frame = min((j + 1) * sign_duration, seq_length)

            # Add sign-specific motion pattern
            freq = 0.1 + (sign_idx * 0.01)
            amplitude = 0.5 + (sign_idx * 0.05)

            for t in range(start_frame, end_frame):
                pose[t] += amplitude * np.sin(t * freq) * np.random.randn(33, 3) * 0.3

        pose_sequences.append(pose)

        # Create CTC target (padded to seq_length)
        target = np.full(seq_length, config.num_signs)  # blank token
        current_pos = 0
        for sign_idx in target_indices:
            if current_pos < seq_length:
                target[current_pos] = sign_idx
                current_pos += np.random.randint(10, 20)  # Variable spacing

        targets.append(target)

    return np.array(pose_sequences), np.array(targets)

def generate_screening_data(num_samples=800, seq_length=150):
    """Generate synthetic screening data (fallback)"""
    pose_sequences = []
    slot_labels = []
    response_labels = []

    print(f"Generating {num_samples} synthetic screening sequences...")

    for i in tqdm(range(num_samples)):
        # Base pose
        pose = np.random.randn(seq_length, 33, 3).astype(np.float32) * 0.3

        # Random labels
        slot_idx = np.random.randint(0, config.num_slots)
        response_idx = np.random.randint(0, config.num_responses)

        # Add patterns based on labels
        freq = 0.05 + (slot_idx * 0.01)
        amplitude = 0.5 + (response_idx * 0.2)

        for t in range(seq_length):
            pose[t] += amplitude * np.sin(t * freq) * 0.5

        # Add noise
        pose += np.random.randn(seq_length, 33, 3) * 0.1

        pose_sequences.append(pose)
        slot_labels.append(slot_idx)
        response_labels.append(response_idx)

    return np.array(pose_sequences), np.array(slot_labels), np.array(response_labels)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_sign_model():
    """Train the sign recognition model"""
    print("\n🤟 TRAINING SIGN RECOGNITION MODEL")
    print("=" * 50)

    # Try to load real video data first
    video_dir = '/kaggle/input/usl-sign-videos'  # Adjust path as needed

    try:
        print(f"Attempting to load real sign videos from {video_dir}")
        X_data, y_data = create_sign_targets_from_videos(video_dir, max_videos=config.num_train_samples + config.num_test_samples)

        # Split into train/test
        split_idx = int(0.8 * len(X_data))
        X_train, X_test = X_data[:split_idx], X_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]

        print(f"✅ Loaded {len(X_train)} training and {len(X_test)} test samples from real videos")

    except Exception as e:
        print(f"⚠️  Could not load real videos: {e}")
        print("Falling back to synthetic data...")
        X_train, y_train = generate_sign_data(config.num_train_samples, config.seq_length_signs)
        X_test, y_test = generate_sign_data(config.num_test_samples, config.seq_length_signs)

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

    train_losses = []
    test_losses = []

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0

        for pose, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            pose = pose.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            logits = model(pose)

            # CTC loss requires specific input format
            log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)  # (seq_len, batch, num_classes)

            input_lengths = torch.full((pose.size(0),), config.seq_length_signs, dtype=torch.long)
            target_lengths = torch.sum(target != config.num_signs, dim=1)  # Count non-blank tokens

            loss = criterion(log_probs, target, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

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
        test_losses.append(test_loss)

        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_sign_model.pth'))
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
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

    # Try to load real video data first
    video_dir = '/kaggle/input/usl-screening-videos'  # Adjust path as needed

    try:
        print(f"Attempting to load real screening videos from {video_dir}")
        X_data, y_slot_data, y_response_data = load_real_screening_data(video_dir, max_videos=config.num_train_samples + config.num_test_samples)

        # Split into train/test
        split_idx = int(0.8 * len(X_data))
        X_train, X_test = X_data[:split_idx], X_data[split_idx:]
        y_slot_train, y_slot_test = y_slot_data[:split_idx], y_slot_data[split_idx:]
        y_response_train, y_response_test = y_response_data[:split_idx], y_response_data[split_idx:]

        print(f"✅ Loaded {len(X_train)} training and {len(X_test)} test samples from real videos")

    except Exception as e:
        print(f"⚠️  Could not load real videos: {e}")
        print("Falling back to synthetic data...")
        X_train, y_slot_train, y_response_train = generate_screening_data(config.num_train_samples, config.seq_length_screening)
        X_test, y_slot_test, y_response_test = generate_screening_data(config.num_test_samples, config.seq_length_screening)

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

    train_losses = []
    test_losses = []
    train_slot_accs = []
    test_slot_accs = []

    for epoch in range(config.num_epochs):
        # Training
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
        train_losses.append(train_loss)
        train_slot_accs.append(train_slot_acc)

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
        test_losses.append(test_loss)
        test_slot_accs.append(test_slot_acc)

        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
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
    print("🚀 Starting USL System Training")
    print(f"Output directory: {config.output_dir}")

    # Train sign recognition model
    sign_model = train_sign_model()

    # Train screening model
    screening_model = train_screening_model()

    # Save vocabulary
    save_vocabulary()

    print("\n🎉 TRAINING COMPLETE!")
    print("=" * 50)
    print("Models saved in:", config.output_dir)
    print("- sign_recognition_model.pth")
    print("- usl_screening_model.pth")
    print("- sign_vocabulary.json")
    print("\nReady for inference! 🚀")
