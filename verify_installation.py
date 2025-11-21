"""
Verification script to check SkinCBM installation and basic functionality.

Run this script to verify that everything is set up correctly:
    python verify_installation.py
"""

import sys
import importlib
from pathlib import Path

def check_module(module_name, description):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description}")
        print(f"  Error: {e}")
        return False

def check_file(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description}")
        return False

def main():
    print("=" * 70)
    print("SkinCBM Installation Verification")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Check Python version
    print("Checking Python version...")
    py_version = sys.version_info
    if py_version >= (3, 8):
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro} (>= 3.8 required)")
    else:
        print(f"✗ Python {py_version.major}.{py_version.minor} (3.8+ required)")
        all_passed = False
    print()
    
    # Check core dependencies
    print("Checking core dependencies...")
    deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
        ("timm", "timm (PyTorch Image Models)"),
    ]
    
    for module, desc in deps:
        if not check_module(module, desc):
            all_passed = False
    print()
    
    # Check SkinCBM modules
    print("Checking SkinCBM modules...")
    skincbm_modules = [
        ("src.models.basic_cbm", "CBM models"),
        ("src.data.base_loader", "Data loaders"),
        ("src.training.trainer", "Training utilities"),
        ("src.utils.information_theory", "Information theory tools"),
    ]
    
    for module, desc in skincbm_modules:
        if not check_module(module, desc):
            all_passed = False
    print()
    
    # Check documentation
    print("Checking documentation...")
    docs = [
        ("README.md", "Main README"),
        ("docs/INSTALLATION.md", "Installation guide"),
        ("docs/QUICKSTART.md", "Quickstart guide"),
        ("docs/ARCHITECTURE.md", "Architecture guide"),
        ("docs/DATASETS.md", "Dataset guide"),
    ]
    
    for filepath, desc in docs:
        if not check_file(filepath, desc):
            all_passed = False
    print()
    
    # Check examples
    print("Checking examples...")
    examples = [
        ("examples/train_basic_cbm.py", "Training script"),
        ("notebooks/01_cbm_training_walkthrough.ipynb", "Tutorial notebook"),
    ]
    
    for filepath, desc in examples:
        if not check_file(filepath, desc):
            all_passed = False
    print()
    
    # Test basic functionality
    print("Testing basic functionality...")
    try:
        import torch
        from src.models.basic_cbm import ConceptBottleneckModel
        
        # Create model
        model = ConceptBottleneckModel(num_concepts=7, num_classes=2)
        print("✓ Model creation")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        concepts, logits = model(dummy_input)
        
        if concepts.shape == (2, 7) and logits.shape == (2, 2):
            print("✓ Forward pass")
        else:
            print("✗ Forward pass (incorrect output shapes)")
            all_passed = False
        
        # Test intervention
        concepts[:, 0] = 1.0
        new_logits = model.predict_from_concepts(concepts)
        if new_logits.shape == (2, 2):
            print("✓ Concept intervention")
        else:
            print("✗ Concept intervention")
            all_passed = False
            
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        all_passed = False
    print()
    
    # Final summary
    print("=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("SkinCBM is correctly installed and ready to use.")
        print()
        print("Next steps:")
        print("  1. Read GETTING_STARTED.md for a quick tutorial")
        print("  2. Check docs/QUICKSTART.md for detailed guide")
        print("  3. Run: python examples/train_basic_cbm.py --help")
        print()
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        print("Please check the errors above and:")
        print("  1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check Python version (3.8+ required)")
        print("  3. Verify you're in the SkinCBM directory")
        print()
        print("For help, see docs/INSTALLATION.md")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
