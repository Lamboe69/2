#!/usr/bin/env python3
"""
USL System - Complete Training Pipeline
Graph-Reasoned Large Vision Models for Ugandan Sign Language Translation
in Infectious Disease Screening

This script provides:
1. Synthetic data generation (or real data loading)
2. Model training with regularization
3. Evaluation and visualization
4. Model checkpointing and inference
"""

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
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    # Data
    num_train_samples = 500
    num_test_samples = 100
    seq_length = 150
    num_joints = 33
    coord_dim = 3
    
    # Model
    hidden_dim = 128
    num_heads = 4
    dropout = 0.3
    
    # Training
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0005
    weight_decay = 0.01
    patience = 15
    
    # Screening
    screening_slots = [
        'symptom_onset', 'fever', 'cough_hemoptysis', 'diarrhea_dehydration',
        'rash', 'exposure', 'travel', 'pregnancy', 'hiv_tb_history', 'danger_signs'
    ]
    responses = ['yes', 'no', 'unknown']
    
    # Paths
    output_dir = './usl_models'
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True)

config = Config()

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_synthetic_pose_data(num_samples=500, seq_length=150):
    """Generate synthetic pose sequences with diversity"""
    pose_sequences = []
    labels_slot = []
    labels_response = []
    
    for i in range(num_samples):
        # Base random pose
        pose = np.random.randn(seq_length, 33, 3).astype(np.float32) * 0.3
        
        # Add structured motion patterns
        slot_idx = i % len(config.screening_slots)
        response_idx = i % len(config.responses)
        
        for t in range(seq_length):
            freq = 0.05 + (slot_idx * 0.01)
            amplitude = 0.5 + (response_idx * 0.2)
            pose[t] = pose[t] + amplitude * np.sin(t * freq) * 0.5
        
        # Add noise
        pose += np.random.randn(seq_length, 33, 3) * 0.1
        
        pose_sequences.append(pose)
        labels_slot.append(slot_idx)
        labels_response.append(response_idx)
    
    return np.array(pose_sequences), np.array(labels_slot), np.array(labels_response)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImprovedPoseModel(nn.Module):
    """Improved pose-based screening model with attention"""
    
    def __init__(self, num_slots=10, num_responses=3, hidden_dim=128):
        super().__init__()
        
        # Pose embedding
        self.joint_embedding = nn.Linear(3, hidden_dim)
        self.embedding_dropout = nn.Dropout(0.3)
        
        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 33,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        # Classification heads
        self.slot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_slots)
        )
        
        self.response_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        """
        Args:
            pose_sequence: (batch, seq_len, 33, 3)
        Returns:
            slot_logits, response_logits, confidence
        """
        batch_size, seq_len, num_joints, coord_dim = pose_sequence.shape
        
        # Embed joints
        pose_embedded = self.joint_embedding(pose_sequence)
        pose_embedded = self.embedding_dropout(pose_embedded)
        
        # Flatten joints
        pose_flat = pose_embedded.reshape(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(pose_flat)
        
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
# TRAINING FUNCTION
# ============================================================================

def train_model(config):
    """Complete training pipeline"""
    
    print("=" * 80)
    print("USL SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # ========================================================================
    # 1. DATA PREPARATION
    # ========================================================================
    
    print("\n📊 Generating synthetic training data...")
    X_train, y_slot_train, y_response_train = generate_synthetic_pose_data(
        config.num_train_samples, config.seq_length
    )
    X_test, y_slot_test, y_response_test = generate_synthetic_pose_data(
        config.num_test_samples, config.seq_length
    )
    
    print(f"✅ Generated {config.num_train_samples} training samples")
    print(f"✅ Generated {config.num_test_samples} test samples")
    print(f"   Shape: {X_train.shape}")
    
    # Create dataloaders
    print("\n📂 Creating dataloaders...")
    
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
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    
    # ========================================================================
    # 2. MODEL INITIALIZATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    model = ImprovedPoseModel(
        num_slots=len(config.screening_slots),
        num_responses=len(config.responses),
        hidden_dim=config.hidden_dim
    ).to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion_slot = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_response = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_confidence = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # ========================================================================
    # 3. TRAINING LOOP
    # ========================================================================
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    test_losses = []
    train_slot_accs = []
    test_slot_accs = []
    train_response_accs = []
    test_response_accs = []
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_slot_correct = 0
        train_response_correct = 0
        train_total = 0
        
        for pose, slot_label, response_label in train_loader:
            pose = pose.to(device)
            slot_label = slot_label.to(device)
            response_label = response_label.to(device)
            
            optimizer.zero_grad()
            
            slot_logits, response_logits, conf_pred = model(pose)
            
            loss_slot = criterion_slot(slot_logits, slot_label)
            loss_response = criterion_response(response_logits, response_label)
            loss_conf = criterion_confidence(conf_pred.squeeze(), torch.rand_like(conf_pred.squeeze()))
            
            loss = loss_slot + loss_response + 0.05 * loss_conf
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            _, slot_pred = torch.max(slot_logits, 1)
            _, response_pred = torch.max(response_logits, 1)
            
            train_slot_correct += (slot_pred == slot_label).sum().item()
            train_response_correct += (response_pred == response_label).sum().item()
            train_total += slot_label.size(0)
        
        train_loss /= len(train_loader)
        train_slot_acc = train_slot_correct / train_total
        train_response_acc = train_response_correct / train_total
        
        train_losses.append(train_loss)
        train_slot_accs.append(train_slot_acc)
        train_response_accs.append(train_response_acc)
        
        # Testing
        model.eval()
        test_loss = 0
        test_slot_correct = 0
        test_response_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for pose, slot_label, response_label in test_loader:
                pose = pose.to(device)
                slot_label = slot_label.to(device)
                response_label = response_label.to(device)
                
                slot_logits, response_logits, conf_pred = model(pose)
                
                loss_slot = criterion_slot(slot_logits, slot_label)
                loss_response = criterion_response(response_logits, response_label)
                loss_conf = criterion_confidence(conf_pred.squeeze(), torch.rand_like(conf_pred.squeeze()))
                
                loss = loss_slot + loss_response + 0.05 * loss_conf
                
                test_loss += loss.item()
                
                _, slot_pred = torch.max(slot_logits, 1)
                _, response_pred = torch.max(response_logits, 1)
                
                test_slot_correct += (slot_pred == slot_label).sum().item()
                test_response_correct += (response_pred == response_label).sum().item()
                test_total += slot_label.size(0)
        
        test_loss /= len(test_loader)
        test_slot_acc = test_slot_correct / test_total
        test_response_acc = test_response_correct / test_total
        
        test_losses.append(test_loss)
        test_slot_accs.append(test_slot_acc)
        test_response_accs.append(test_response_acc)
        
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Slot Acc: {train_slot_acc:.4f} | Response Acc: {train_response_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f} | Slot Acc: {test_slot_acc:.4f} | Response Acc: {test_response_acc:.4f}")
            print(f"  Patience: {patience_counter}/{config.patience}")
        
        if patience_counter >= config.patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    print("\n✅ Training completed!")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_model.pth')))
    
    # ========================================================================
    # 4. EVALUATION & VISUALIZATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2, alpha=0.7)
    axes[0, 0].plot(test_losses, label='Test Loss', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Slot accuracy
    axes[0, 1].plot(train_slot_accs, label='Train Slot Acc', linewidth=2, alpha=0.7)
    axes[0, 1].plot(test_slot_accs, label='Test Slot Acc', linewidth=2, alpha=0.7)
    axes[0, 1].axhline(y=1/len(config.screening_slots), color='r', linestyle='--', label='Random Baseline')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Screening Slot Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Response accuracy
    axes[1, 0].plot(train_response_accs, label='Train Response Acc', linewidth=2, alpha=0.7)
    axes[1, 0].plot(test_response_accs, label='Test Response Acc', linewidth=2, alpha=0.7)
    axes[1, 0].axhline(y=1/len(config.responses), color='r', linestyle='--', label='Random Baseline')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Response Classification Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final metrics
    final_metrics = {
        'Metric': ['Train Loss', 'Test Loss', 'Train Slot Acc', 'Test Slot Acc', 
                   'Train Response Acc', 'Test Response Acc'],
        'Value': [train_losses[-1], test_losses[-1], train_slot_accs[-1], test_slot_accs[-1],
                  train_response_accs[-1], test_response_accs[-1]]
    }
    
    metrics_text = "\n".join([f"{m}: {v:.4f}" for m, v in zip(final_metrics['Metric'], final_metrics['Value'])])
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Final Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'training_results.png'), dpi=100, bbox_inches='tight')
    print(f"\n📊 Training plot saved to {config.output_dir}/training_results.png")
    
    print(f"\n✅ Final Test Slot Accuracy: {test_slot_accs[-1]:.4f}")
    print(f"✅ Final Test Response Accuracy: {test_response_accs[-1]:.4f}")
    print(f"✅ Final Test Loss: {test_losses[-1]:.4f}")
    
    # ========================================================================
    # 5. SAVE MODEL
    # ========================================================================
    
    print("\n💾 Saving model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_slots': len(config.screening_slots),
            'num_responses': len(config.responses),
            'hidden_dim': config.hidden_dim,
            'screening_slots': config.screening_slots,
            'responses': config.responses
        }
    }, os.path.join(config.output_dir, 'usl_screening_model.pth'))
    
    print(f"✅ Model saved to {config.output_dir}/usl_screening_model.pth")
    
    # ========================================================================
    # 6. INFERENCE EXAMPLE
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("INFERENCE EXAMPLE")
    print("=" * 80)
    
    model.eval()
    
    for i in range(3):
        sample_pose = torch.FloatTensor(np.random.randn(1, 150, 33, 3)).to(device)
        
        with torch.no_grad():
            slot_logits, response_logits, confidence = model(sample_pose)
            
            slot_pred = torch.argmax(slot_logits, dim=1).item()
            response_pred = torch.argmax(response_logits, dim=1).item()
            conf = confidence.item()
            
            print(f"\n📋 Sample {i+1} Prediction:")
            print(f"   Screening Slot: {config.screening_slots[slot_pred]}")
            print(f"   Response: {config.responses[response_pred]}")
            print(f"   Confidence: {conf:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    
    return model, config

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    model, config = train_model(config)
