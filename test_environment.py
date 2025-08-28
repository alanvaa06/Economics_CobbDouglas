"""
Environment diagnostic script to test package imports.
"""

import sys
import os

print("=== ENVIRONMENT DIAGNOSTIC ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

print("\n=== TESTING IMPORTS ===")
packages = ['numpy', 'pandas', 'matplotlib', 'scipy', 'ipywidgets']

for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package}: {version}")
    except ImportError as e:
        print(f"❌ {package}: {e}")

print("\n=== TESTING PROJECT IMPORTS ===")
try:
    from src.economics_models import SolowModel, SolowParameters
    print("✅ Project imports: SUCCESS")
    
    # Quick test
    params = SolowParameters()
    model = SolowModel(params)
    growth_rate = model.get_final_growth_rate()
    print(f"✅ Model test: Growth rate = {growth_rate:.3f}%")
    
except Exception as e:
    print(f"❌ Project imports: {e}")

print("\n=== RECOMMENDATION ===")
print("If all packages show ✅, run: python main.py")
print("If packages show ❌, check your environment activation")

