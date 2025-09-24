#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed
and the inference module can be imported correctly.
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported."""
    required_modules = [
        'mmcv',
        'mmseg',
        'mmengine', 
        'cv2',
        'numpy',
        'pandas',
        'skimage',
        'slidingwindow',
        'PIL',
        'matplotlib'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
            failed_imports.append(module)
    
    return failed_imports

def test_local_modules():
    """Test if local modules can be imported."""
    local_modules = [
        'utils',
        'quantify_seg_results'
    ]
    
    failed_imports = []
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
            failed_imports.append(module)
    
    return failed_imports

def main():
    """Main test function."""
    print("Testing Crack Detection Inference Setup...")
    print("=" * 50)
    
    # Test external dependencies
    print("\n1. Testing external dependencies:")
    external_failures = test_imports()
    
    # Test local modules
    print("\n2. Testing local modules:")
    local_failures = test_local_modules()
    
    # Summary
    print("\n" + "=" * 50)
    if not external_failures and not local_failures:
        print("✓ All tests passed! Setup is ready.")
        return 0
    else:
        print("✗ Some tests failed:")
        if external_failures:
            print(f"  - External modules: {', '.join(external_failures)}")
        if local_failures:
            print(f"  - Local modules: {', '.join(local_failures)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

