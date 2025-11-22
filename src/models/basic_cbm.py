"""
Basic Concept Bottleneck Model Implementation

This module provides a clean, educational implementation of CBMs with:
- Concept encoder (image → concepts)
- Task predictor (concepts → diagnosis)
- Intervention capabilities
- Uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import timm


class ConceptEncoder(nn.Module):
    """
    Encodes images into concept predictions.
    
    Architecture: Pretrained CNN → Concept prediction heads
    
    Args:
        backbone: Pretrained vision model name (e.g., 'resnet50')
        num_concepts: Number of concepts to predict
        concept_types: List of 'binary' or 'continuous' for each concept
        freeze_backbone: Whether to freeze pretrained weights
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_concepts: int = 23,
        num_classes_per_concept: int = 3,
        concept_types: Optional[List[str]] = None,
        freeze_backbone: bool = False,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.num_classes_per_concept = num_classes_per_concept
        self.concept_types = concept_types or ['multiclass'] * num_concepts
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy_input).shape[1]
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Concept prediction heads - each outputs num_classes_per_concept logits
        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes_per_concept)
            )
            for _ in range(num_concepts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict concepts from images.
        
        Args:
            x: Images [batch_size, 3, H, W]
            
        Returns:
            concepts: [batch_size, num_concepts * num_classes_per_concept]
                Softmax probabilities for each concept class
                Reshaped to [batch_size, num_concepts, num_classes_per_concept]
        """
        # Extract features
        features = self.backbone(x)
        
        # Predict each concept
        concept_probs = []
        for i, head in enumerate(self.concept_heads):
            logits = head(features)  # [batch, num_classes_per_concept]
            probs = F.softmax(logits, dim=1)  # Convert to probabilities
            concept_probs.append(probs)
        
        # Stack into [batch, num_concepts, num_classes_per_concept]
        concepts = torch.stack(concept_probs, dim=1)
        # Flatten to [batch, num_concepts * num_classes_per_concept]
        concepts = concepts.view(concepts.size(0), -1)
        return concepts


class TaskPredictor(nn.Module):
    """
    Predicts task labels from concepts using a simple linear layer.
    
    Most interpretable architecture: y = w^T c + b
    Each weight shows how much each concept contributes to the prediction.
    
    Args:
        num_concepts: Number of input concepts
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        num_concepts: int,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        # Simple linear layer: most interpretable
        self.predictor = nn.Linear(num_concepts, num_classes)
    
    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Predict task labels from concepts.
        
        Args:
            concepts: [batch_size, num_concepts]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.predictor(concepts)
    
    def get_concept_importance(self) -> torch.Tensor:
        """
        Get concept importance weights.
        
        Returns:
            weights: [num_concepts, num_classes]
        """
        return self.predictor.weight.detach()


class ConceptBottleneckModel(nn.Module):
    """
    Complete Concept Bottleneck Model.
    
    Two-stage architecture:
    1. Image → Concepts (ConceptEncoder)
    2. Concepts → Task Prediction (TaskPredictor)
    
    Supports:
    - Joint training (end-to-end)
    - Sequential training (concepts first, then task)
    - Concept intervention (modify concept values during inference)
    
    Args:
        num_concepts: Number of concepts
        num_classes: Number of task classes
        concept_types: List of 'binary' or 'continuous' for each concept
        backbone: Pretrained vision model name
        task_architecture: Task predictor type (only 'linear' supported in this basic implementation)
        freeze_backbone: Whether to freeze CNN backbone
    
    Example:
        >>> model = ConceptBottleneckModel(num_concepts=23, num_classes=2)
        >>> concepts, logits = model(images)
        >>> 
        >>> # Intervene on concept 5
        >>> concepts[:, 5] = 1.0
        >>> corrected_logits = model.predict_from_concepts(concepts)
    """
    
    def __init__(
        self,
        num_concepts: int = 23,
        num_classes: int = 2,
        num_classes_per_concept: int = 3,
        concept_types: Optional[List[str]] = None,
        backbone: str = "resnet50",
        freeze_backbone: bool = False,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.num_classes_per_concept = num_classes_per_concept
        
        # Concept encoder
        self.concept_encoder = ConceptEncoder(
            backbone=backbone,
            num_concepts=num_concepts,
            num_classes_per_concept=num_classes_per_concept,
            concept_types=concept_types,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained
        )
        
        # Task predictor (simple linear layer)
        # Input size is num_concepts * num_classes_per_concept (flattened concept probabilities)
        self.task_predictor = TaskPredictor(
            num_concepts=num_concepts * num_classes_per_concept,
            num_classes=num_classes
        )
    
    def forward(
        self,
        x: torch.Tensor,
        concepts: Optional[torch.Tensor] = None,
        return_concepts: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional concept intervention.
        
        Args:
            x: Images [batch_size, 3, H, W]
            concepts: Optional ground-truth or intervened concepts [batch_size, num_concepts]
            return_concepts: Whether to return concept predictions
            
        Returns:
            If return_concepts=True: (concepts, task_logits)
            If return_concepts=False: task_logits only
        """
        # Predict concepts from images (unless provided)
        if concepts is None:
            concepts = self.concept_encoder(x)
        
        # Predict task from concepts
        task_logits = self.task_predictor(concepts)
        
        if return_concepts:
            return concepts, task_logits
        else:
            return task_logits
    
    def predict_from_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Predict task labels directly from concept values.
        
        Used for concept intervention: modify concepts, then re-predict.
        
        Args:
            concepts: [batch_size, num_concepts]
            
        Returns:
            task_logits: [batch_size, num_classes]
        """
        return self.task_predictor(concepts)
    
    def intervene(
        self,
        x: torch.Tensor,
        intervention_idxs: List[int],
        intervention_values: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform concept intervention on predictions.
        
        Args:
            x: Images [batch_size, 3, H, W]
            intervention_idxs: Concept indices to intervene on
            intervention_values: New values for intervened concepts
            
        Returns:
            original_concepts: Concepts before intervention
            intervened_concepts: Concepts after intervention
            intervened_logits: Task predictions after intervention
        """
        # Get original predictions
        original_concepts = self.concept_encoder(x)
        
        # Create intervened concepts
        intervened_concepts = original_concepts.clone()
        for idx, value in zip(intervention_idxs, intervention_values):
            intervened_concepts[:, idx] = value
        
        # Re-predict with intervened concepts
        intervened_logits = self.predict_from_concepts(intervened_concepts)
        
        return original_concepts, intervened_concepts, intervened_logits
    
    def get_concept_importance(self) -> Optional[torch.Tensor]:
        """
        Get concept importance weights (if using linear task predictor).
        
        Returns:
            weights: [num_concepts, num_classes] or None
        """
        try:
            return self.task_predictor.get_concept_importance()
        except NotImplementedError:
            return None
    
    def freeze_concepts(self):
        """Freeze concept encoder (for sequential training)."""
        for param in self.concept_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_concepts(self):
        """Unfreeze concept encoder."""
        for param in self.concept_encoder.parameters():
            param.requires_grad = True
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_concepts': self.num_concepts,
            'num_classes': self.num_classes,
            'num_classes_per_concept': self.num_classes_per_concept,
            'concept_types': self.concept_encoder.concept_types,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            num_concepts=checkpoint['num_concepts'],
            num_classes=checkpoint['num_classes'],
            num_classes_per_concept=checkpoint.get('num_classes_per_concept', 3),  # Default to 3 for new models
            concept_types=checkpoint['concept_types'],
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)


def create_cbm_from_config(config: Dict) -> ConceptBottleneckModel:
    """
    Create CBM from configuration dictionary.
    
    Args:
        config: Dictionary with model hyperparameters
        
    Returns:
        Initialized CBM model
    """
    return ConceptBottleneckModel(
        num_concepts=config.get('num_concepts', 23),
        num_classes=config.get('num_classes', 2),
        num_classes_per_concept=config.get('num_classes_per_concept', 3),
        concept_types=config.get('concept_types', None),
        backbone=config.get('backbone', 'resnet50'),
        freeze_backbone=config.get('freeze_backbone', False),
        pretrained=config.get('pretrained', True)
    )
