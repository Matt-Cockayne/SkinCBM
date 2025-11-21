#!/bin/bash
# Quick test to verify training setup before submitting to HPC

echo "======================================"
echo "Pre-Flight Check for SkinCBM Training"
echo "======================================"
echo ""

# Check 1: Data directory exists
echo "[1/6] Checking dataset..."
DATA_PATH="/home/xrai/datasets/derm7pt/release_v0"
if [ -d "$DATA_PATH" ]; then
    IMG_COUNT=$(ls "$DATA_PATH/images/"*.jpg 2>/dev/null | wc -l)
    echo "✓ Dataset found: $DATA_PATH"
    echo "  Images: $IMG_COUNT files"
    if [ -f "$DATA_PATH/meta/meta.csv" ]; then
        echo "  ✓ Metadata CSV found"
    else
        echo "  ✗ WARNING: meta/meta.csv not found"
    fi
else
    echo "✗ ERROR: Dataset not found at $DATA_PATH"
    exit 1
fi
echo ""

# Check 2: Python environment
echo "[2/6] Checking Python environment..."
if command -v python3 &> /dev/null; then
    echo "✓ Python3: $(python3 --version)"
else
    echo "✗ ERROR: python3 not found"
    exit 1
fi
echo ""

# Check 3: Required packages
echo "[3/6] Checking required packages..."
REQUIRED_PACKAGES=("torch" "torchvision" "timm" "pandas" "numpy" "PIL")
ALL_FOUND=true

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
        echo "  ✓ $pkg ($VERSION)"
    else
        echo "  ✗ $pkg NOT FOUND"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo ""
    echo "Some packages are missing. Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi
echo ""

# Check 4: SkinCBM code
echo "[4/6] Checking SkinCBM code..."
if [ -f "src/models/basic_cbm.py" ]; then
    echo "✓ Core model found"
else
    echo "✗ ERROR: src/models/basic_cbm.py not found"
    exit 1
fi

if [ -f "src/data/derm7pt_adapter.py" ]; then
    echo "✓ Data adapter found"
else
    echo "✗ ERROR: src/data/derm7pt_adapter.py not found"
    exit 1
fi

if [ -f "examples/train_basic_cbm.py" ]; then
    echo "✓ Training script found"
else
    echo "✗ ERROR: examples/train_basic_cbm.py not found"
    exit 1
fi
echo ""

# Check 5: Can import the code
echo "[5/6] Testing imports..."
if python3 -c "import sys; sys.path.insert(0, '.'); from src.models.basic_cbm import ConceptBottleneckModel" 2>/dev/null; then
    echo "✓ Can import ConceptBottleneckModel"
else
    echo "✗ ERROR: Cannot import ConceptBottleneckModel"
    echo "  Check for syntax errors in src/models/basic_cbm.py"
    exit 1
fi

if python3 -c "import sys; sys.path.insert(0, '.'); from src.data.derm7pt_adapter import create_derm7pt_dataloaders" 2>/dev/null; then
    echo "✓ Can import derm7pt_adapter"
else
    echo "✗ ERROR: Cannot import derm7pt_adapter"
    echo "  Check for syntax errors or missing dependencies"
    exit 1
fi
echo ""

# Check 6: SLURM script exists
echo "[6/6] Checking SLURM script..."
if [ -f "train_cbm.slurm" ]; then
    echo "✓ SLURM script found: train_cbm.slurm"
    if [ -x "train_cbm.slurm" ]; then
        echo "  ✓ Script is executable"
    else
        echo "  ⚠ Script not executable (not required for sbatch)"
    fi
else
    echo "✗ ERROR: train_cbm.slurm not found"
    exit 1
fi
echo ""

# Summary
echo "======================================"
echo "✅ Pre-flight checks PASSED"
echo "======================================"
echo ""
echo "Ready to submit training job!"
echo ""
echo "To submit:"
echo "  sbatch train_cbm.slurm"
echo ""
echo "To monitor:"
echo "  squeue -u \$USER"
echo "  tail -f skincbm_train_*.out"
echo ""
echo "Expected completion: ~10 minutes on V100"
echo "Output location: trained_models/derm7pt_best/"
echo ""
