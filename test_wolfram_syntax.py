#!/usr/bin/env python
"""Test script to check wolfram_utils.py syntax"""

import sys
import traceback

try:
    from src.core import wolfram_utils
    print("✓ Syntax OK - wolfram_utils module imported successfully")
    print(f"✓ query_wolfram_alpha function available: {hasattr(wolfram_utils, 'query_wolfram_alpha')}")
    print(f"✓ extract_wolfram_plaintext_answer function available: {hasattr(wolfram_utils, 'extract_wolfram_plaintext_answer')}")
except Exception as e:
    print(f"✗ Error importing wolfram_utils: {e}")
    traceback.print_exc()
    sys.exit(1)
