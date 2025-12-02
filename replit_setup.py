#!/usr/bin/env python3
"""
Replit Setup Script for USL Screening System
Handles model storage and dependency management for Replit environment
"""

import os
import subprocess
import sys

def check_storage():
    """Check current storage usage"""
    try:
        result = subprocess.run(['du', '-sh', '.'], capture_output=True, text=True)
        print(f"Current directory size: {result.stdout.strip()}")

        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        print("Disk usage:")
        print(result.stdout)
    except Exception as e:
        print(f"Storage check error: {e}")

def check_models():
    """Check if model files exist and their sizes"""
    model_dir = "usl_models"
    if os.path.exists(model_dir):
        print(f"\nModel directory found: {model_dir}")
        total_size = 0
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                print(".1f")

        print(".1f"
    else:
        print(f"\nModel directory NOT found: {model_dir}")

def install_deps():
    """Install dependencies with error handling"""
    print("\nInstalling dependencies...")

    # Install in smaller batches to avoid timeouts
    deps = [
        "streamlit",
        "torch torchvision",
        "numpy pandas",
        "plotly tqdm pillow",
        "matplotlib",
        "opencv-python-headless",
        "mediapipe protobuf",
        "psutil"
    ]

    for dep in deps:
        print(f"Installing: {dep}")
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', dep],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✓ Successfully installed: {dep}")
            else:
                print(f"✗ Failed to install: {dep}")
                print(result.stderr)
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout installing: {dep}")
        except Exception as e:
            print(f"❌ Error installing {dep}: {e}")

def main():
    print("🔧 USL Screening System - Replit Setup")
    print("=" * 50)

    check_storage()
    check_models()

    print("\n🚀 Starting dependency installation...")
    install_deps()

    print("\n✅ Setup complete!")
    print("Click the 'Run' button to start your Streamlit app!")

if __name__ == "__main__":
    main()
