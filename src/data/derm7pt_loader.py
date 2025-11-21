"""
Derm7pt Dataset Loader

The 7-point checklist dataset for melanoma diagnosis.

Dataset structure expected:
data/derm7pt/
├── images/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── concepts.csv  # Concept annotations
└── labels.csv    # Diagnosis labels

Concepts (7-point checklist):
1. Atypical pigment network
2. Blue-whitish veil
3. Atypical vascular pattern
4. Irregular streaks
5. Irregular pigmentation
6. Irregular dots and globules
7. Regression structures
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional
import numpy as np
from torchvision import transforms

from src.data.base_loader import BaseDataLoader


class Derm7ptDataset(BaseDataLoader):
    """
    Derm7pt dataset with 7-point checklist concepts.
    
    Args:
        data_path: Root directory containing derm7pt folder
        split: 'train', 'val', or 'test'
        transform: Optional image transformations
        train_val_test_split: Tuple of (train_frac, val_frac, test_frac)
    """
    
    # 7-point checklist concept names
    CONCEPT_NAMES = [
        "atypical_pigment_network",
        "blue_whitish_veil",
        "atypical_vascular_pattern",
        "irregular_streaks",
        "irregular_pigmentation",
        "irregular_dots_globules",
        "regression_structures"
    ]
    
    CLASS_NAMES = ["Nevus", "Melanoma"]
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 42
    ):
        self.data_path = os.path.join(data_path, 'derm7pt')
        self.split = split
        self.num_classes = 2
        
        # Default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Load data
        self._load_data(train_val_test_split, random_seed)
    
    def _load_data(self, split_ratios: Tuple[float, float, float], seed: int):
        """Load and split the dataset."""
        
        # Check if processed data exists
        concepts_path = os.path.join(self.data_path, 'concepts.csv')
        labels_path = os.path.join(self.data_path, 'labels.csv')
        
        if not os.path.exists(concepts_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}\n"
                f"Please download and prepare the Derm7pt dataset.\n"
                f"See docs/DATASETS.md for instructions."
            )
        
        # Load annotations
        concepts_df = pd.read_csv(concepts_path)
        labels_df = pd.read_csv(labels_path)
        
        # Merge on image ID
        data_df = pd.merge(concepts_df, labels_df, on='image_id')
        
        # Split data
        np.random.seed(seed)
        n_samples = len(data_df)
        indices = np.random.permutation(n_samples)
        
        train_size = int(split_ratios[0] * n_samples)
        val_size = int(split_ratios[1] * n_samples)
        
        if self.split == 'train':
            split_indices = indices[:train_size]
        elif self.split == 'val':
            split_indices = indices[train_size:train_size + val_size]
        else:  # test
            split_indices = indices[train_size + val_size:]
        
        self.data = data_df.iloc[split_indices].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} {self.split} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample.
        
        Returns:
            image: [3, 224, 224] tensor
            concepts: [7] binary tensor
            label: 0 (Nevus) or 1 (Melanoma)
        """
        row = self.data.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.data_path, 'images', row['image_id'] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Load concepts (7 binary values)
        concepts = torch.tensor([
            row[concept_name] for concept_name in self.CONCEPT_NAMES
        ], dtype=torch.float32)
        
        # Load label
        label = int(row['diagnosis'])  # 0 or 1
        
        return image, concepts, label
    
    def get_concept_names(self) -> List[str]:
        return self.CONCEPT_NAMES
    
    def get_class_names(self) -> List[str]:
        return self.CLASS_NAMES


def create_derm7pt_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_seed: int = 42
):
    """
    Create Derm7pt dataloaders.
    
    Args:
        data_path: Root directory containing derm7pt folder
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_val_test_split: Split ratios
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from src.data.base_loader import create_dataloaders
    
    train_dataset = Derm7ptDataset(
        data_path,
        split='train',
        train_val_test_split=train_val_test_split,
        random_seed=random_seed
    )
    
    val_dataset = Derm7ptDataset(
        data_path,
        split='val',
        train_val_test_split=train_val_test_split,
        random_seed=random_seed
    )
    
    test_dataset = Derm7ptDataset(
        data_path,
        split='test',
        train_val_test_split=train_val_test_split,
        random_seed=random_seed
    )
    
    return create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
