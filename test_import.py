#!/usr/bin/env python3
"""Test individual imports to find the issue"""

print("Testing imports...")

try:
    import os
    print("OK: os imported")
except Exception as e:
    print(f"FAIL: os failed: {e}")

try:
    import torch
    print("OK: torch imported")
except Exception as e:
    print(f"FAIL: torch failed: {e}")

try:
    import torch.nn as nn
    print("OK: torch.nn imported")
except Exception as e:
    print(f"FAIL: torch.nn failed: {e}")

try:
    import numpy as np
    print("OK: numpy imported")
except Exception as e:
    print(f"FAIL: numpy failed: {e}")

try:
    import cv2
    print("OK: cv2 imported")
except Exception as e:
    print(f"FAIL: cv2 failed: {e}")

try:
    import json
    print("OK: json imported")
except Exception as e:
    print(f"FAIL: json failed: {e}")

try:
    import mediapipe as mp
    print("OK: mediapipe imported")
except Exception as e:
    print(f"FAIL: mediapipe failed: {e}")

try:
    from datetime import datetime
    print("OK: datetime imported")
except Exception as e:
    print(f"FAIL: datetime failed: {e}")

print("All basic imports tested.")
