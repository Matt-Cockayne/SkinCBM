# Installation Guide

This guide will help you set up SkinCBM on your system.

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: Optional but recommended for GPU acceleration (CUDA 11.7+ with PyTorch 2.0+)
- **Storage**: ~5GB for dependencies + datasets

## Quick Installation

```bash
# Navigate to SkinCBM directory
cd SkinCBM

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.models.basic_cbm import ConceptBottleneckModel; print('✓ Installation successful!')"
```

## Detailed Installation Steps

### 1. Python Environment

We recommend using a virtual environment to avoid dependency conflicts:

**Using venv:**
```bash
python -m venv skincbm_env
source skincbm_env/bin/activate  # Linux/Mac
# OR
skincbm_env\Scripts\activate  # Windows
```

**Using conda:**
```bash
conda create -n skincbm python=3.10
conda activate skincbm
```

### 2. Install PyTorch

Install PyTorch with CUDA support (if available):

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

Verify PyTorch installation:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **Data processing**: numpy, pandas, pillow, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Information theory**: scipy
- **Model architectures**: timm (PyTorch Image Models)
- **Notebooks**: jupyter, ipywidgets
- **Utilities**: tqdm, tensorboard, pyyaml

### 4. Verify Installation

Run the test script:

```bash
python -c "
from src.models.basic_cbm import ConceptBottleneckModel
from src.data.base_loader import BaseDataLoader
from src.utils.information_theory import compute_mutual_information
import torch

# Test model creation
model = ConceptBottleneckModel(num_concepts=7, num_classes=2)
print('✓ Model creation successful')

# Test forward pass
dummy_input = torch.randn(2, 3, 224, 224)
concepts, logits = model(dummy_input)
print(f'✓ Forward pass successful: {concepts.shape}, {logits.shape}')

print('\n🎉 Installation verified!')
"
```

## Download Datasets

See [DATASETS.md](DATASETS.md) for instructions on downloading and preparing:
- Derm7pt dataset
- SkinCon dataset
- Optional: Chest X-ray, diabetic retinopathy datasets

## GPU Setup (Optional)

### Check GPU Availability

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Troubleshooting GPU Issues

**CUDA not detected:**
```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory errors:**
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use gradient checkpointing (implemented in model)
- Use mixed precision training (FP16)

## Optional: Development Setup

For contributing or advanced usage:

```bash
# Install in editable mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black src/ examples/
```

## Common Installation Issues

### Issue: ImportError for timm

**Solution:**
```bash
pip install timm>=0.9.0
```

### Issue: Jupyter notebooks not loading

**Solution:**
```bash
pip install jupyter ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Issue: scikit-learn version conflicts

**Solution:**
```bash
pip install --upgrade scikit-learn>=1.3.0
```

### Issue: Permission errors on Linux/Mac

**Solution:**
```bash
pip install --user -r requirements.txt
```

## Uninstallation

```bash
# Remove virtual environment
deactivate  # Exit environment first
rm -rf venv/  # or skincbm_env/

# Or with conda
conda deactivate
conda env remove -n skincbm
```

## Next Steps

After installation:
1. **[Quickstart Guide](QUICKSTART.md)** - Train your first CBM in 5 minutes
2. **[Architecture Guide](ARCHITECTURE.md)** - Understand the CBM framework
3. **[Example Notebooks](../notebooks/)** - Interactive tutorials

## Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: Read [docs/README.md](README.md)
- **Examples**: Check [examples/](../examples/) for reference code

---

**Estimated Installation Time**: 5-10 minutes (depending on internet speed)
