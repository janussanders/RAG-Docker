#!/usr/bin/env python3

import pytest
import sys
from pathlib import Path

def main():
    """Run all tests"""
    # Add src to Python path
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.append(str(src_path))
    
    # Run pytest with verbosity
    pytest.main(['-v', '--capture=no'])

if __name__ == "__main__":
    main() 