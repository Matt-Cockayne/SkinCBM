# SkinCBM Data Download Instructions

This file contains instructions for obtaining the datasets used in SkinCBM.

## Quick Start

The datasets are not included in this repository. You need to:
1. Request access from official sources
2. Download and extract to `data/` directory
3. Follow the preprocessing instructions below

## Directory Structure

After setup, your `data/` folder should look like:

```
data/
├── derm7pt/
│   ├── images/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   ├── concepts.csv
│   └── labels.csv
│
└── skincon/  (optional)
    ├── images/
    ├── concepts.json
    └── labels.json
```

## Dataset: Derm7pt

**TODO**: Add download links and preprocessing scripts once dataset access is obtained.

For now, see [docs/DATASETS.md](../docs/DATASETS.md) for complete information.

## Creating Demo Data

For testing the code without real data, you can generate synthetic data:

```python
import os
import numpy as np
from PIL import Image
import pandas as pd

# Create directories
os.makedirs('data/derm7pt/images', exist_ok=True)

# Generate 100 synthetic images
for i in range(100):
    # Random image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(img).save(f'data/derm7pt/images/{i:03d}.jpg')

# Generate synthetic concepts
concepts_data = {
    'image_id': [f'{i:03d}' for i in range(100)],
    'atypical_pigment_network': np.random.randint(0, 2, 100),
    'blue_whitish_veil': np.random.randint(0, 2, 100),
    'atypical_vascular_pattern': np.random.randint(0, 2, 100),
    'irregular_streaks': np.random.randint(0, 2, 100),
    'irregular_pigmentation': np.random.randint(0, 2, 100),
    'irregular_dots_globules': np.random.randint(0, 2, 100),
    'regression_structures': np.random.randint(0, 2, 100),
}
pd.DataFrame(concepts_data).to_csv('data/derm7pt/concepts.csv', index=False)

# Generate synthetic labels
labels_data = {
    'image_id': [f'{i:03d}' for i in range(100)],
    'diagnosis': np.random.randint(0, 2, 100)
}
pd.DataFrame(labels_data).to_csv('data/derm7pt/labels.csv', index=False)

print("Synthetic demo data created!")
```

**Note**: This is only for testing the code. Real medical data is needed for actual research.
