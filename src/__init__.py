"""
SkinCBM: Concept Bottleneck Models for Medical Image Diagnosis

A comprehensive educational framework for understanding and implementing
interpretable medical image diagnosis using Concept Bottleneck Models.
"""

__version__ = "1.0.0"
__author__ = "Matthew J. Cockayne"

from src.models.basic_cbm import ConceptBottleneckModel
from src.data.base_loader import BaseDataLoader

__all__ = [
    "ConceptBottleneckModel",
    "BaseDataLoader",
]
