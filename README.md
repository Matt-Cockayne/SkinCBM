# SkinCBM: Concept Bottleneck Models for Interpretable Skin Lesion Diagnosis

An implementation of Concept Bottleneck Models (CBMs) applied to dermoscopic image classification using the 7-point checklist protocol. CBMs enforce an interpretable intermediate representation — predicted clinical concepts — through which all diagnostic reasoning must pass, enabling both model transparency and clinician-guided concept intervention at inference time.

This work builds on [Koh et al. (2020)](https://arxiv.org/abs/2007.04612) and adapts CBMs to the dermatology domain using the Derm7pt dataset.

## Method Overview

A CBM decomposes classification into two stages:

```
Image --> Concept Encoder --> Concept Predictions --> Task Predictor --> Diagnosis Prediction
          (e.g. ResNet-50)    (7-point checklist)     (Linear layer)
```

1. **Concept Encoder**: A pretrained ResNet-50 backbone with per-concept classification heads predicts the 7-point checklist attributes (pigment network, blue-whitish veil, vascular structures, streaks, pigmentation, dots/globules, regression structures).
2. **Task Predictor**: A linear layer maps the predicted concept vector to a binary diagnosis (melanoma vs. non-melanoma). The linear architecture preserves full weight interpretability.

Because the task predictor operates only on concept predictions, clinicians can *intervene* by correcting individual concept values and observing the effect on the diagnosis — a property unique to bottleneck architectures.

Three training strategies are supported:
- **Joint**: End-to-end optimisation of concept and task losses.
- **Sequential**: Concept encoder trained first, then frozen while the task predictor is trained.
- **Independent**: Concept encoder and task predictor trained separately with ground-truth concepts.

## Repository Structure

```
SkinCBM/
├── src/
│   ├── models/
│   │   └── basic_cbm.py              # CBM model (encoder + predictor)
│   ├── data/
│   │   ├── base_loader.py            # Abstract dataset interface
│   │   ├── derm7pt_loader.py         # Derm7pt dataset loader
│   │   └── derm7pt_adapter.py        # Adapter for derm7pt package
│   ├── training/
│   │   └── trainer.py                # Training loop and evaluation
│   └── utils/
│       └── visualization.py          # Concept and intervention plots
│
├── examples/
│   ├── train_basic_cbm.py            # Training script
│   ├── demo_sample_data.py           # Inference on sample cases
│   ├── demo_intervention.py          # Concept intervention demo
│   └── intervention_analysis.py      # Systematic intervention evaluation
│
├── notebooks/
│   ├── 01_cbm_training_walkthrough.ipynb
│   ├── 02_demo_with_sample_data.ipynb
│   ├── 03_demo_intervention.ipynb
│   └── sample_data_derm7pt/          # Included sample cases
│
├── docs/
│   ├── INSTALLATION.md
│   ├── QUICKSTART.md
│   ├── ARCHITECTURE.md
│   └── DATASETS.md
│
└── data/
    └── derm7pt/                      # External derm7pt dataloader package
```

## Installation

Requires Python 3.8+ and PyTorch 2.0+.

```bash
cd SkinCBM
pip install -r requirements.txt
python verify_installation.py
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for environment setup details and GPU configuration.

## Usage

### Inference Demo (no dataset required)

Four sample dermoscopy cases are included for immediate use:

```bash
python3 examples/demo_sample_data.py
python3 examples/demo_intervention.py
```

Or via notebooks:
```bash
jupyter notebook notebooks/02_demo_with_sample_data.ipynb
jupyter notebook notebooks/03_demo_intervention.ipynb
```

### Training

```bash
python3 examples/train_basic_cbm.py \
    --data_path /path/to/derm7pt/release_v0 \
    --epochs 50 \
    --output_dir ./outputs/my_cbm
```

For HPC/SLURM submission, see [HPC_TRAINING.md](HPC_TRAINING.md).

### Concept Intervention

After training, concept values can be overridden at inference time:

```python
from src.models.basic_cbm import ConceptBottleneckModel

model = ConceptBottleneckModel.load("outputs/best_model.pth")
concepts, logits = model(image)

# Override a concept prediction
concepts[:, 2] = 1.0
corrected_logits = model.predict_from_concepts(concepts)
```

## Dataset

This implementation uses the **Derm7pt** dataset (~2,000 dermoscopy images with 7-point checklist annotations).

- **Source**: [https://derm.cs.sfu.ca/](https://derm.cs.sfu.ca/) (academic access required)
- **Concepts**: 7 clinical attributes, each with 3 ordinal classes (absent / regular / irregular)
- **Task**: Binary classification (melanoma vs. non-melanoma)

See [docs/DATASETS.md](docs/DATASETS.md) for download and preparation instructions.

## Results

Performance on Derm7pt (joint training, ResNet-50 backbone):

| Epochs | Concept Accuracy | Task F1 | Training Time (V100) |
|--------|-----------------|---------|---------------------|
| 20     | 70--75%         | 65--70% | ~5 min              |
| 50     | 75--80%         | 68--72% | ~10 min             |
| 100    | 75--82%         | 70--75% | ~20 min             |

The interpretability--accuracy trade-off is approximately 5 percentage points relative to a comparable black-box classifier, consistent with findings in Koh et al. (2020).

## Documentation

- [INSTALLATION.md](docs/INSTALLATION.md) -- Environment setup
- [QUICKSTART.md](docs/QUICKSTART.md) -- Training walkthrough
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) -- Model design and training strategies
- [DATASETS.md](docs/DATASETS.md) -- Data acquisition and format

## Citation

```bibtex
@software{skincbm2025,
  title={SkinCBM: Concept Bottleneck Models for Skin Lesion Diagnosis},
  author={Cockayne, Matthew J.},
  year={2025},
  url={https://github.com/Matt-Cockayne/SynergyCBM/tree/main/SkinCBM}
}
```

## References

- Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). Concept Bottleneck Models. *ICML*. [arXiv:2007.04612](https://arxiv.org/abs/2007.04612)
- Kawahara, J., Daneshvar, S., Argenziano, G., & Hamarneh, G. (2019). Seven-Point Checklist and Skin Lesion Classification using Multitask Multimodal Neural Nets. *IEEE JBHI*.

## License

MIT License -- see [LICENSE](LICENSE).
