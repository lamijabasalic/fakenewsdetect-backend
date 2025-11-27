#!/usr/bin/env python
"""Test script to diagnose server startup issues"""
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

print("Testing imports...")
try:
    from preprocessing import preprocess_bosnian_text
    print("✓ preprocessing imported")
except Exception as e:
    print(f"✗ preprocessing import failed: {e}")
    sys.exit(1)

try:
    from main import app
    print("✓ main app imported")
except Exception as e:
    print(f"✗ main app import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful! Server should start correctly.")

