#!/usr/bin/env python3
"""
Run script for Image Pyramid Blending

Usage:
    python run.py
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main
from src.main import main

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
