"""
Adapter for the derm7pt dataset to work with our CBM framework.

This wraps the existing derm7pt dataloader to provide a consistent interface.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

# Add parent 'data' directory to path so derm7pt can be imported as a package
data_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')),
    '/home/csc29/projects/SynergyCBM/SkinCBM/data'
]

DERM7PT_AVAILABLE = False
for data_path in data_paths:
    derm7pt_dir = os.path.join(data_path, 'derm7pt')
    if os.path.exists(derm7pt_dir) and os.path.exists(os.path.join(derm7pt_dir, 'dataset.py')):
        sys.path.insert(0, data_path)
        try:
            from derm7pt.dataset import Derm7PtDatasetGroupInfrequent
            import pandas as pd
            DERM7PT_AVAILABLE = True
            print(f"✓ Loaded derm7pt module from: {data_path}")
            break
        except ImportError as e:
            print(f"Warning: Failed to import from {data_path}: {e}")
            continue

if not DERM7PT_AVAILABLE:
    print("Warning: derm7pt dataloader not found. Using sample data only.")


class Derm7ptCBMAdapter(Dataset):
    """
    Adapter for derm7pt dataset to work with CBM framework.
    
    Converts 7-point checklist features into binary concept labels.
    """
    
    # 7-point checklist - main categories (not sub-labels)
    CONCEPT_NAMES = [
        "pigment_network",
        "blue_whitish_veil", 
        "vascular_structures",
        "streaks",
        "pigmentation",
        "dots_and_globules",
        "regression_structures"
    ]
    
    CLASS_NAMES = ["Benign", "Malignant"]
    
    def __init__(
        self,
        data_path: str = "/home/xrai/datasets/derm7pt/release_v0",
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            data_path: Path to derm7pt release_v0 directory
            split: 'train', 'valid', or 'test'
            image_size: Size to resize images to
        """
        if not DERM7PT_AVAILABLE:
            raise ImportError(
                "derm7pt dataset module not found. "
                "Make sure the derm7pt dataloader is installed at data/derm7pt/"
            )
        
        self.data_path = data_path
        self.split = 'valid' if split == 'val' else split  # Map 'val' to 'valid'
        self.image_size = image_size
        
        # Load the derm7pt dataset
        self.derm_data, self.columns = self._load_dataset()
        
        # Create the underlying dataset
        from derm7pt.dataloader import dataset as Derm7ptDataset
        self.dataset = Derm7ptDataset(
            derm=self.derm_data,
            shape=image_size,
            mode=self.split
        )
        
        # Get diagnosis labels (for binary classification)
        # DIAG is the abbreviation for diagnosis in the derm7pt dataset
        self.diagnosis_labels = self.derm_data.get_labels(
            data_type=self.split, 
            one_hot=False
        )['DIAG']
        
        # Get concept labels from 7-point checklist
        self.concept_labels = self._extract_concepts()
        
        print(f"Loaded {len(self)} {split} samples from derm7pt")
    
    def _load_dataset(self):
        """Load derm7pt dataset from release directory."""
        dir_meta = os.path.join(self.data_path, 'meta')
        dir_images = os.path.join(self.data_path, 'images')

        meta_df = pd.read_csv(os.path.join(dir_meta, 'meta.csv'))
        
        columns_of_interest = [
            'level_of_diagnostic_difficulty', 'elevation', 'location', 'sex', 'management'
        ]

        train_indexes = list(pd.read_csv(os.path.join(dir_meta, 'train_indexes.csv'))['indexes'])
        valid_indexes = list(pd.read_csv(os.path.join(dir_meta, 'valid_indexes.csv'))['indexes'])
        test_indexes = list(pd.read_csv(os.path.join(dir_meta, 'test_indexes.csv'))['indexes'])

        derm_data_group = Derm7PtDatasetGroupInfrequent(
            dir_images=dir_images,
            metadata_df=meta_df.copy(),
            train_indexes=train_indexes,
            valid_indexes=valid_indexes,
            test_indexes=test_indexes
        )

        derm_data_group.meta_train = derm_data_group.meta_train[columns_of_interest]
        derm_data_group.meta_valid = derm_data_group.meta_valid[columns_of_interest]
        derm_data_group.meta_test = derm_data_group.meta_test[columns_of_interest]
        
        return derm_data_group, columns_of_interest
    
    def _extract_concepts(self) -> np.ndarray:
        """
        Extract concept labels from 7-point checklist.
        
        Gets the numeric labels for each of the 7 main categories.
        These are already processed in the dataset as *_numeric columns.
        """
        # Get the appropriate dataframe split
        if self.split == 'train':
            df = self.derm_data.train
        elif self.split == 'valid':
            df = self.derm_data.valid
        else:
            df = self.derm_data.test
        
        # Extract the numeric concept labels (0, 1, 2, etc. for each category)
        # These represent the different sub-labels within each category
        concepts = np.zeros((len(df), 7), dtype=np.float32)
        
        concept_columns = [
            'pigment_network_numeric',
            'blue_whitish_veil_numeric',
            'vascular_structures_numeric',
            'streaks_numeric',
            'pigmentation_numeric',
            'dots_and_globules_numeric',
            'regression_structures_numeric'
        ]
        
        for i, col in enumerate(concept_columns):
            concepts[:, i] = df[col].values
        
        return concepts
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample.
        
        Returns:
            image: [3, 224, 224] tensor (dermoscopic image)
            concepts: [7] binary tensor
            label: 0 (Benign) or 1 (Malignant)
        """
        try:
            # Get images and labels from underlying dataset
            # Returns: (dermoscopy_img, clinic_img, metadata, meta_con, [DIAG, PN, BWV, VS, PIG, STR, DaG, RS])
            derm_img, clinic_img, metadata, meta_con, concept_labels = self.dataset[idx]
            
            # Use dermoscopic image (more relevant for diagnosis)
            image = derm_img
            
            # Extract concept labels from the list of tensors
            # concept_labels = [DIAG, PN, BWV, VS, PIG, STR, DaG, RS]
            # We want PN through RS (skip DIAG as it's the task label)
            concepts = torch.stack([
                concept_labels[1],  # PN
                concept_labels[2],  # BWV
                concept_labels[3],  # VS
                concept_labels[4],  # PIG
                concept_labels[5],  # STR
                concept_labels[6],  # DaG
                concept_labels[7],  # RS
            ]).squeeze().float()
            
            # Get diagnosis label and convert to binary melanoma classification
            # Derm7PtDatasetGroupInfrequent has 5 classes (0-4), with melanoma=2
            # Convert to binary: 0=non-melanoma, 1=melanoma
            diag_value = concept_labels[0].item()
            label = 1 if diag_value == 2 else 0
            
            return image, concepts, label
            
        except Exception as e:
            # If image loading fails, skip to next sample
            print(f"Warning: Failed to load sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def get_concept_names(self) -> List[str]:
        return self.CONCEPT_NAMES
    
    def get_class_names(self) -> List[str]:
        return self.CLASS_NAMES


def create_derm7pt_dataloaders(
    data_path: str = "/home/xrai/datasets/derm7pt/release_v0",
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    use_class_weights: bool = True
):
    """
    Create dataloaders for derm7pt dataset with class balancing for melanoma detection.
    
    Args:
        data_path: Path to derm7pt release_v0 directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Size to resize images to
        use_class_weights: Whether to use weighted sampling for class balance
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = Derm7ptCBMAdapter(data_path, split='train', image_size=image_size)
    val_dataset = Derm7ptCBMAdapter(data_path, split='val', image_size=image_size)
    test_dataset = Derm7ptCBMAdapter(data_path, split='test', image_size=image_size)
    
    # Calculate class weights for balanced sampling
    sampler = None
    if use_class_weights:
        # Count melanoma vs non-melanoma in training set
        labels = []
        for i in range(len(train_dataset)):
            try:
                _, _, label = train_dataset[i]
                labels.append(label)
            except:
                continue
        
        labels = torch.tensor(labels)
        class_counts = torch.bincount(labels)
        
        # Calculate weights: inverse frequency
        weights = 1.0 / class_counts.float()
        sample_weights = weights[labels]
        
        print(f"Class distribution - Non-melanoma: {class_counts[0]}, Melanoma: {class_counts[1]}")
        print(f"Using weighted sampling with weights: Non-melanoma={weights[0]:.4f}, Melanoma={weights[1]:.4f}")
        
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
