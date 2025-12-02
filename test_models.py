#!/usr/bin/env python3
"""Test script to verify USL models load correctly"""

try:
    print("Testing basic import...")
    import usl_inference
    print("OK: usl_inference module imported successfully")

    print("Testing class import...")
    from usl_inference import USLInferencePipeline
    print("OK: USLInferencePipeline class imported successfully")

except Exception as e:
    print(f"FAIL: Import Error: {e}")
    import traceback
    traceback.print_exc()
