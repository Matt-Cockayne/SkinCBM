"""
Base data loader interface for CBM datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np


class BaseDataLoader(Dataset, ABC):
    """
    Abstract base class for CBM datasets.
    
    All dataset loaders should implement:
    - __len__: Return dataset size
    - __getitem__: Return (image, concepts, label)
    - get_concept_names: Return list of concept names
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            image: [3, H, W] tensor
            concepts: [num_concepts] tensor (binary or continuous)
            label: Integer class label
        """
        pass
    
    @abstractmethod
    def get_concept_names(self) -> List[str]:
        """Return list of concept names."""
        pass
    
    def get_concept_types(self) -> List[str]:
        """
        Return concept types ('binary' or 'continuous').
        Default: all binary.
        """
        return ['binary'] * len(self.get_concept_names())
    
    def get_class_names(self) -> List[str]:
        """Return list of class names."""
        return [f"Class_{i}" for i in range(self.num_classes)]


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
