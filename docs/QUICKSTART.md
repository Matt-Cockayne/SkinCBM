# Quickstart Guide

Train your first Concept Bottleneck Model in 5 minutes!

## Prerequisites

- Completed [installation](INSTALLATION.md)
- Downloaded Derm7pt dataset (see [DATASETS.md](DATASETS.md))

## 5-Minute Training

### Step 1: Verify Installation

```bash
cd SkinCBM
python -c "from src.models.basic_cbm import ConceptBottleneckModel; print('✓ Ready')"
```

### Step 2: Prepare Data

Place your Derm7pt dataset in this structure:
```
data/derm7pt/
├── images/
│   ├── 001.jpg
│   └── ...
├── concepts.csv
└── labels.csv
```

### Step 3: Train!

```bash
python examples/train_basic_cbm.py \
    --dataset derm7pt \
    --data_path ./data \
    --epochs 20 \
    --output_dir ./outputs/my_first_cbm
```

**Training time**: ~10-15 minutes on GPU, ~1 hour on CPU

### Step 4: Check Results

```bash
# View results
cat outputs/my_first_cbm/results.txt

# Output:
# Test Results:
#   Concept Accuracy: 0.7845
#   Task Accuracy:    0.7231
#   Task F1 Score:    0.7156
```

## Understanding Your Results

### Concept Accuracy (78%)
- How well the model predicts individual concepts
- **Good**: >75%
- **Excellent**: >85%

### Task Accuracy (72%)
- Final diagnosis accuracy
- Compares to ~75-78% for black-box models
- **Trade-off**: ~5% accuracy for full interpretability

### Task F1 Score (72%)
- Balanced measure accounting for class imbalance
- More reliable than accuracy for medical tasks

## Try Concept Intervention

```python
import torch
from src.models.basic_cbm import ConceptBottleneckModel
from PIL import Image
from torchvision import transforms

# Load your trained model
model = ConceptBottleneckModel.load('outputs/my_first_cbm/best_model.pth')
model.eval()

# Load an image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = transform(Image.open('data/derm7pt/images/001.jpg')).unsqueeze(0)

# Predict
concepts, logits = model(image)
prediction = logits.argmax(dim=1).item()

print(f"Original prediction: {prediction}")
print(f"Concepts: {concepts.squeeze().tolist()}")

# Intervene: Fix concept 2 (blue-whitish veil)
concepts[0, 2] = 1.0  # Set to present
new_logits = model.predict_from_concepts(concepts)
new_prediction = new_logits.argmax(dim=1).item()

print(f"After intervention: {new_prediction}")
```

## Next Steps

### 1. Explore Interactively
Open the Jupyter notebooks:
```bash
jupyter notebook notebooks/01_cbm_training_walkthrough.ipynb
```

### 2. Train with More Epochs

```bash
# Quick test (10 epochs)
python examples/train_basic_cbm.py \
    --epochs 10 \
    --data_path ./data \
    --output_dir ./outputs/quick_test

# Full training (50 epochs)
python examples/train_basic_cbm.py \
    --epochs 50 \
    --data_path ./data \
    --output_dir ./outputs/full_training
```

### 3. Adjust Concept-Task Balance

Control how much the model focuses on concepts vs final task:

```bash
# More emphasis on concepts
python examples/train_basic_cbm.py \
    --concept_loss_weight 2.0 \
    --data_path ./data \
    --output_dir ./outputs/cbm_concept_heavy

# More emphasis on task
python examples/train_basic_cbm.py \
    --concept_loss_weight 0.5 \
    --data_path ./data \
    --output_dir ./outputs/cbm_task_heavy
```

### 4. Use Different Backbones

```bash
# EfficientNet (faster, fewer parameters)
python examples/train_basic_cbm.py \
    --backbone efficientnet_b0 \
    --data_path ./data \
    --output_dir ./outputs/cbm_efficientnet

# ResNet-101 (more capacity)
python examples/train_basic_cbm.py \
    --backbone resnet101 \
    --data_path ./data \
    --output_dir ./outputs/cbm_resnet101
```

## Common Issues

### Out of Memory
```bash
# Reduce batch size
python examples/train_basic_cbm.py --batch_size 8 ...
```

### Slow Training
```bash
# Reduce epochs for quick experimentation
python examples/train_basic_cbm.py --epochs 10 ...
```

### Poor Concept Accuracy
- Check dataset quality
- Increase `--concept_loss_weight`
- Try sequential training (see [TRAINING.md](TRAINING.md))

### Poor Task Accuracy
- Increase training epochs (try 50-100)
- Adjust concept loss weight
- Check for high synergy in information analysis (may need non-linear predictor)

## Learning Resources

1. **[Architecture Guide](ARCHITECTURE.md)** - Deep dive into CBM design
2. **[Intervention Tutorial](INTERVENTION.md)** - Master concept intervention
3. **[Interactive Notebooks](../notebooks/)** - Hands-on experiments

## Benchmarking Your Model

Expected performance on Derm7pt:

| Epochs | Concept Acc | Task F1 | Training Time |
|--------|-------------|---------|---------------|
| 20 | 70-75% | 65-70% | 5 min |
| 50 | 75-80% | 68-72% | 10 min |
| 100 | 75-82% | 70-75% | 20 min |

*Times on V100 GPU with default settings*

## Getting Help

- **Examples**: Check [examples/](../examples/) for more scripts
- **Documentation**: Read [docs/](.) for detailed guides
- **Issues**: Report problems on GitHub

---

**🎉 Congratulations!** You've trained your first interpretable medical AI model.

Now explore the [notebooks](../notebooks/) to see what your model learned!
