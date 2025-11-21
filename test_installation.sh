#!/bin/bash
# Quick test script to verify the SkinCBM repository works

echo "=================================="
echo "SkinCBM Repository Test"
echo "=================================="
echo ""

# Test 1: Check imports
echo "[1/3] Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
from src.models.basic_cbm import ConceptBottleneckModel
from src.training.trainer import CBMTrainer
from src.utils.information_theory import compute_mutual_information
print('✓ All imports successful')
" || { echo "✗ Import test failed"; exit 1; }

echo ""

# Test 2: Run sample data demo
echo "[2/3] Running sample data demo..."
python3 examples/demo_sample_data.py > /dev/null 2>&1 || { echo "✗ Demo failed"; exit 1; }
echo "✓ Demo completed successfully"
echo ""

# Test 3: Check outputs
echo "[3/3] Checking generated outputs..."
if [ -f "outputs/demo_case_1.png" ] && [ -f "outputs/demo_case_577.png" ] && [ -f "outputs/demo_case_578.png" ]; then
    echo "✓ All visualization files created"
else
    echo "✗ Some output files missing"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ All tests passed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Try the interactive demo:"
echo "     jupyter notebook notebooks/02_demo_with_sample_data.ipynb"
echo ""
echo "  2. Train on full dataset:"
echo "     python3 examples/train_basic_cbm.py --data_path /home/xrai/datasets/derm7pt/release_v0 --epochs 50"
echo ""
