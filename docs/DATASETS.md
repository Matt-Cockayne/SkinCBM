# Dataset Information

This document provides information about supported datasets and how to obtain them.

## Derm7pt Dataset

### Overview
- **Task**: Melanoma vs Nevus classification
- **Images**: ~2,000 dermoscopy images
- **Concepts**: 7-point checklist for melanoma diagnosis
- **Format**: JPG images + CSV annotations

### 7-Point Checklist Concepts

1. **Atypical pigment network**: Irregular network of pigmented lines
2. **Blue-whitish veil**: Blue-white overlay obscuring skin
3. **Atypical vascular pattern**: Irregular blood vessel patterns
4. **Irregular streaks**: Radial streaming at lesion periphery
5. **Irregular pigmentation**: Uneven coloration
6. **Irregular dots and globules**: Varying-sized dots/globules
7. **Regression structures**: White/blue areas indicating regression

### Download Instructions

**Option 1: Official Source**
1. Visit: https://derm.cs.sfu.ca/
2. Request access (academic use)
3. Download dataset
4. Extract to `data/derm7pt/`

**Option 2: Prepare Your Own**

If you have dermoscopy images with 7-point checklist annotations:

```bash
# Create directory structure
mkdir -p data/derm7pt/images

# Organize your data:
data/derm7pt/
├── images/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── concepts.csv
└── labels.csv
```

**concepts.csv format**:
```csv
image_id,atypical_pigment_network,blue_whitish_veil,atypical_vascular_pattern,irregular_streaks,irregular_pigmentation,irregular_dots_globules,regression_structures
001,1,0,1,1,0,1,0
002,0,1,0,0,1,1,1
...
```

**labels.csv format**:
```csv
image_id,diagnosis
001,1
002,0
...
```
- 0 = Nevus (benign)
- 1 = Melanoma (malignant)

### Expected Performance

| Training | Concept Acc | Task F1 |
|----------|-------------|---------|
| 20 epochs | 70-75% | 65-70% |
| 50 epochs | 75-80% | 68-72% |
| 100 epochs | 75-82% | 70-75% |

## SkinCon Dataset (Coming Soon)

### Overview
- **Task**: Multi-class skin condition classification
- **Images**: ~10,000 clinical images
- **Concepts**: 23 morphological attributes
- **Format**: JPG images + JSON annotations

### Concepts

**Primary Morphology** (7):
- Nodule, Ulcer, Papule, Plaque, Pustule, Bulla, Patch

**Secondary Features** (5):
- Scale, Crust, Erosion, Hyperpigmentation, Scar

**Additional Attributes** (11):
- Border irregularity, Asymmetry, Color variation, etc.

### Download Instructions

(To be added - dataset in preparation)

## Optional: Extending to Other Domains

### Chest X-Ray (CheXpert)

**Task**: Multi-label pathology detection
**Concepts**: 14 pathology labels (Cardiomegaly, Edema, etc.)

```python
# Adapt the data loader
from src.data.base_loader import BaseDataLoader

class CheXpertDataset(BaseDataLoader):
    CONCEPT_NAMES = [
        "no_finding", "enlarged_cardiomediastinum", "cardiomegaly",
        "lung_opacity", "lung_lesion", "edema", "consolidation",
        "pneumonia", "atelectasis", "pneumothorax", "pleural_effusion",
        "pleural_other", "fracture", "support_devices"
    ]
    # ... implement __getitem__, etc.
```

### Diabetic Retinopathy

**Task**: DR severity grading
**Concepts**: Lesion types (microaneurysms, exudates, hemorrhages)

```python
class DiabeticRetinopathyDataset(BaseDataLoader):
    CONCEPT_NAMES = [
        "microaneurysms", "hemorrhages", "hard_exudates",
        "soft_exudates", "neovascularization", "fibrous_proliferation"
    ]
    # ... implement dataset loading
```

## Dataset Statistics

### Recommended Split Ratios

```python
# Default split
train_val_test_split = (0.7, 0.15, 0.15)

# Small dataset (< 1000 samples)
train_val_test_split = (0.6, 0.2, 0.2)

# Large dataset (> 10000 samples)
train_val_test_split = (0.8, 0.1, 0.1)
```

### Minimum Sample Sizes

For reliable CBM training:
- **Minimum**: 500 samples
- **Recommended**: 1,000+ samples
- **Ideal**: 5,000+ samples

For concept-specific requirements:
- Each concept should have 50+ positive and negative examples
- Imbalanced concepts (< 5% prevalence) may need special handling

## Data Augmentation

Default augmentations for skin lesion images:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

**Recommended for medical images**:
- ✓ Horizontal/vertical flips (lesions have no orientation)
- ✓ Rotation (small angles, ±20°)
- ✓ Color jitter (imaging condition variation)
- ✗ Aggressive crops (may remove diagnostic features)
- ✗ Extreme rotations (unrealistic)

## Creating Your Own Dataset

### Step 1: Define Concepts

Choose concepts that are:
1. **Human-interpretable**: Clear, understandable attributes
2. **Diagnostically relevant**: Important for the task
3. **Reliably annotatable**: Experts can label consistently
4. **Complete**: Cover all decision factors
5. **Independent**: Minimal overlap between concepts

### Step 2: Collect Annotations

```python
# Example annotation interface (pseudo-code)
for image in dataset:
    display(image)
    
    for concept in concepts:
        label = expert_annotates(concept)
        annotations[image][concept] = label
    
    diagnosis = expert_annotates("final_diagnosis")
    labels[image] = diagnosis
```

### Step 3: Validate Quality

```python
# Check inter-rater reliability
from sklearn.metrics import cohen_kappa_score

# Have multiple experts label same images
kappa = cohen_kappa_score(expert1_labels, expert2_labels)
# kappa > 0.6: Good agreement
# kappa > 0.8: Excellent agreement
```

### Step 4: Create Data Loader

```python
from src.data.base_loader import BaseDataLoader

class MyDataset(BaseDataLoader):
    CONCEPT_NAMES = ["concept1", "concept2", ...]
    CLASS_NAMES = ["class1", "class2", ...]
    
    def __init__(self, data_path, split='train'):
        # Load and split data
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, concepts, label = self.load_sample(idx)
        return image, concepts, label
```

## Troubleshooting

### Issue: Poor Concept Accuracy

**Solutions**:
- Check annotation quality (inter-rater agreement)
- Increase concept loss weight
- Use sequential training strategy
- Collect more training data

### Issue: Class Imbalance

**Solutions**:
```python
# Weighted loss
from torch.nn import CrossEntropyLoss

class_weights = torch.tensor([1.0, 3.0])  # Weight minority class higher
loss_fn = CrossEntropyLoss(weight=class_weights)
```

### Issue: Rare Concepts

**Solutions**:
- Use focal loss for hard-to-learn concepts
- Oversample images with rare concepts
- Consider removing concepts with <5% prevalence

## License and Attribution

When using public datasets:
1. Cite original papers
2. Follow usage terms
3. Acknowledge data sources
4. Respect privacy/ethics guidelines

---

**Need help?** Open an issue or check the [FAQ](FAQ.md)
