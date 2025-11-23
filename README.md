# SkinCBM: Concept Bottleneck Models for Interpretable Medical Diagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, educational implementation of Concept Bottleneck Models (CBMs) for skin cancer diagnosis. This repository demonstrates how to build interpretable neural networks that reason through human-understandable concepts.

## 🎯 What is a Concept Bottleneck Model?

CBMs are neural networks with a two-stage architecture that forces all reasoning through interpretable concepts:

```
Image → Concepts → Diagnosis Prediction
         ↓
    Interpretable!
    (can intervene here)
```

**Example**: Instead of directly predicting "Melanoma", the model first identifies:
- Irregular border: 92%
- Asymmetric shape: 78%
- Blue-white veil: 15% ← **You can correct this!**
- Multiple colors: 88%

Then uses these concepts to make the final diagnosis.

## 🚀 Quick Start

### Installation

```bash
# Clone and install
cd SkinCBM
pip install -r requirements.txt
```

### Quick Demo with Sample Data (No dataset required!)

Try the CBM on 3 sample dermoscopy images:

```bash
# Run inference demo
python3 examples/demo_sample_data.py

# Test concept intervention
python3 examples/demo_intervention.py

# Or use the interactive notebooks
jupyter notebook notebooks/02_demo_with_sample_data.ipynb
jupyter notebook notebooks/03_demo_intervention.ipynb
```

This will show you:
- How concepts work
- Concept intervention
- Model interpretability

### Train on Full Dataset

**Local/Interactive:**
```bash
# Train on full derm7pt dataset
python3 examples/train_basic_cbm.py \
    --data_path /home/xrai/datasets/derm7pt/release_v0 \
    --epochs 50 \
    --output_dir ./outputs/my_cbm
```

**HPC/SLURM:**
```bash
# Submit training job to GPU queue
sbatch train_cbm.slurm

# Monitor progress
tail -f skincbm_train_*.out
```

See [HPC_TRAINING.md](HPC_TRAINING.md) for details.

### Use the Trained Model

```python
from src.models.basic_cbm import ConceptBottleneckModel
import torch

# Load model
model = ConceptBottleneckModel.load("outputs/my_first_cbm/best_model.pth")

# Make prediction
concepts, logits = model(image)
prediction = logits.argmax(dim=1)

# Intervene on concepts
concepts[:, 2] = 1.0  # Correct a wrong concept
new_logits = model.predict_from_concepts(concepts)
```

## 📓 Interactive Tutorial

**Quick Demo** (no dataset needed):
```bash
jupyter notebook notebooks/02_demo_with_sample_data.ipynb
```

This notebook uses 3 sample cases to demonstrate:
1. Loading dermoscopy images
2. Understanding the 7-point checklist
3. Running CBM inference
4. Concept intervention
5. Interpreting model weights

**Full Training Walkthrough** (requires full dataset):
```bash
jupyter notebook notebooks/01_cbm_training_walkthrough.ipynb
```

## 🏗️ Repository Structure

```
SkinCBM/
├── src/
│   ├── models/
│   │   └── basic_cbm.py              # Core CBM implementation
│   ├── data/
│   │   ├── base_loader.py            # Abstract dataset interface
│   │   └── derm7pt_loader.py         # Derm7pt dataset loader
│   ├── training/
│   │   └── trainer.py                # Training utilities
│   └── utils/
│       └── information_theory.py     # MI, synergy, completeness
│
├── examples/
│   ├── train_basic_cbm.py            # Train on full dataset
│   ├── demo_sample_data.py           # Quick demo with 3 samples
│   ├── demo_intervention.py          # Concept intervention examples
│   └── intervention_analysis.py      # Systematic intervention analysis
│
├── notebooks/
│   ├── 01_cbm_training_walkthrough.ipynb  # Full training tutorial
│   ├── 02_demo_with_sample_data.ipynb     # Quick demo (no dataset needed!)
│   ├── 03_demo_intervention.ipynb         # Concept intervention demo
│   └── sample_data_derm7pt/               # 3 sample cases with metadata
│
└── docs/                             # Detailed documentation
    ├── INSTALLATION.md
    ├── QUICKSTART.md
    ├── ARCHITECTURE.md
    └── DATASETS.md
```

## 📚 Key Features

### 1. Clean CBM Implementation

```python
model = ConceptBottleneckModel(
    num_concepts=7,
    num_classes=2,
    backbone='resnet50',       # Pretrained encoder
    task_architecture='linear' # Interpretable predictor
)
```

### 2. Concept Intervention

```python
# Original prediction
concepts, logits = model(image)
print(f"Prediction: {logits.argmax()}")  # Benign

# Fix wrong concept
concepts[:, 5] = 1.0  # "Irregular border" should be present
corrected_logits = model.predict_from_concepts(concepts)
print(f"New prediction: {corrected_logits.argmax()}")  # Malignant
```

### 3. Information-Theoretic Analysis

```python
from src.utils.information_theory import analyze_cbm_information

# Compute MI, synergy, completeness
results = analyze_cbm_information(model, dataloader)
print(f"Concept completeness: {results['completeness']:.2f}")
print(f"Synergy: {results['synergy']:.3f} bits")
```

### 4. Flexible Training

- **Joint training**: Train concepts and task predictor together (default)
- **Sequential training**: Train concepts first, then task predictor
- **Independent training**: Train concepts only (use external predictor)

## 🔬 Datasets

### Sample Data (Included!)

The repository includes 3 sample cases in `notebooks/sample_data_derm7pt/`:
- Case 1: Basal Cell Carcinoma
- Case 577: Melanoma  
- Case 578: Melanoma (high 7-point score)

Perfect for quick testing and demos without needing the full dataset!

### Derm7pt (Full Dataset)

- **Size**: ~2,000 dermoscopy images
- **Concepts**: 7-point checklist for melanoma diagnosis
- **Task**: Binary classification (melanoma vs nevus)
- **Location**: `/home/xrai/datasets/derm7pt/release_v0`
- **Download**: See [docs/DATASETS.md](docs/DATASETS.md)

### Adding Your Own Dataset

```python
from src.data.base_loader import BaseDataLoader

class MyDataset(BaseDataLoader):
    CONCEPT_NAMES = ["concept1", "concept2", ...]
    CLASS_NAMES = ["class1", "class2"]
    
    def __getitem__(self, idx):
        image = load_image(idx)
        concepts = get_concept_labels(idx)
        label = get_class_label(idx)
        return image, concepts, label
```

## 📊 Expected Performance

On Derm7pt dataset:

| Training | Concept Acc | Task F1 | Time (V100) |
|----------|-------------|---------|-------------|
| 20 epochs | 70-75% | 65-70% | ~5 min |
| 50 epochs | 75-80% | 68-72% | ~10 min |
| 100 epochs | 75-82% | 70-75% | ~20 min |

**Trade-off**: ~5% accuracy vs black-box models, but gain full interpretability + intervention capability.

## 📖 Documentation

For detailed guides, see the `docs/` folder:

- **[INSTALLATION.md](docs/INSTALLATION.md)** - Setup instructions
- **[QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute tutorial
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Model design details
- **[DATASETS.md](docs/DATASETS.md)** - Data loading and preparation

## 🎓 Why This Repository?

**Educational Focus**: Every design decision is explained with clear documentation

**Production-Ready Code**: Clean, modular, well-tested implementation

**Information Theory Integration**: Novel analysis tools for concept quality

**Complete Pipeline**: Data loading → training → evaluation → intervention

## 🤝 Contributing

This is an educational repository. Contributions welcome:
- Bug fixes or improvements
- New dataset loaders
- Additional documentation or examples

## 📚 Citation

```bibtex
@software{skincbm2025,
  title={SkinCBM: Concept Bottleneck Models for Medical Diagnosis},
  author={Cockayne, Matthew J.},
  year={2025},
  url={https://github.com/Matt-Cockayne/SynergyCBM/tree/main/SkinCBM}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🔗 References

- **Original CBM Paper**: [Koh et al., 2020](https://arxiv.org/abs/2007.04612)
- **Information Theory**: Cover & Thomas, "Elements of Information Theory"
- **Medical AI Interpretability**: [Nature Medicine Review](https://www.nature.com/articles/s41591-021-01614-0)

---

**Last Updated**: November 2025
