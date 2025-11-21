"""
Training utilities for Concept Bottleneck Models.

Supports:
- Joint training (end-to-end)
- Sequential training (concepts first, then task predictor)
- Concept intervention during training
- Multi-stage curriculum learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Tuple
import numpy as np
from tqdm import tqdm
import os


class CBMTrainer:
    """
    Trainer for Concept Bottleneck Models.
    
    Supports multiple training strategies:
    - 'joint': Train concepts and task predictor together
    - 'sequential': Train concepts first, then freeze and train task predictor
    - 'independent': Train concepts and task separately
    
    Args:
        model: CBM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        learning_rate: Learning rate
        concept_loss_weight: Weight for concept loss (vs task loss)
        training_strategy: 'joint', 'sequential', or 'independent'
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        concept_loss_weight: float = 1.0,
        training_strategy: str = 'joint'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.training_strategy = training_strategy
        
        # Loss functions
        self.concept_loss_fn = nn.BCEWithLogitsLoss()
        self.task_loss_fn = nn.CrossEntropyLoss()
        self.concept_loss_weight = concept_loss_weight
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Tracking
        self.train_history = {'concept_loss': [], 'task_loss': [], 'total_loss': []}
        self.val_history = {'concept_acc': [], 'task_acc': [], 'task_f1': []}
        self.best_val_acc = 0.0
    
    def train_epoch(self, epoch: int, use_ground_truth_concepts: bool = False) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            use_ground_truth_concepts: Use GT concepts for task predictor (intervention)
            
        Returns:
            metrics: Dictionary with loss values
        """
        self.model.train()
        
        total_concept_loss = 0.0
        total_task_loss = 0.0
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for images, gt_concepts, labels in pbar:
            images = images.to(self.device)
            gt_concepts = gt_concepts.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if use_ground_truth_concepts:
                # Train task predictor with perfect concepts (intervention simulation)
                pred_concepts, task_logits = self.model(images, concepts=gt_concepts)
            else:
                # Normal forward pass
                pred_concepts, task_logits = self.model(images)
            
            # Compute losses
            # Concept loss (binary cross-entropy)
            concept_loss = self.concept_loss_fn(pred_concepts, gt_concepts)
            
            # Task loss (cross-entropy)
            task_loss = self.task_loss_fn(task_logits, labels)
            
            # Combined loss
            if self.training_strategy == 'independent':
                # Only train concepts or task based on current phase
                loss = concept_loss if epoch < 10 else task_loss
            else:
                loss = self.concept_loss_weight * concept_loss + task_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_concept_loss += concept_loss.item()
            total_task_loss += task_loss.item()
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'c_loss': f"{concept_loss.item():.3f}",
                't_loss': f"{task_loss.item():.3f}"
            })
        
        # Average losses
        metrics = {
            'concept_loss': total_concept_loss / n_batches,
            'task_loss': total_task_loss / n_batches,
            'total_loss': total_loss / n_batches
        }
        
        return metrics
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Returns:
            metrics: Dictionary with accuracy and F1 scores
        """
        if dataloader is None:
            dataloader = self.val_loader
        
        self.model.eval()
        
        all_concept_preds = []
        all_concept_targets = []
        all_task_preds = []
        all_task_targets = []
        
        with torch.no_grad():
            for images, gt_concepts, labels in dataloader:
                images = images.to(self.device)
                gt_concepts = gt_concepts.to(self.device)
                
                # Forward pass
                pred_concepts, task_logits = self.model(images)
                
                # Collect predictions
                all_concept_preds.append((pred_concepts > 0.5).cpu().numpy())
                all_concept_targets.append(gt_concepts.cpu().numpy())
                all_task_preds.append(task_logits.argmax(dim=1).cpu().numpy())
                all_task_targets.append(labels.numpy())
        
        # Concatenate
        all_concept_preds = np.vstack(all_concept_preds)
        all_concept_targets = np.vstack(all_concept_targets)
        all_task_preds = np.concatenate(all_task_preds)
        all_task_targets = np.concatenate(all_task_targets)
        
        # Compute metrics
        concept_acc = (all_concept_preds == all_concept_targets).mean()
        task_acc = (all_task_preds == all_task_targets).mean()
        
        # F1 score
        from sklearn.metrics import f1_score
        task_f1 = f1_score(all_task_targets, all_task_preds, average='binary' if len(np.unique(all_task_targets)) == 2 else 'macro')
        
        metrics = {
            'concept_acc': concept_acc,
            'task_acc': task_acc,
            'task_f1': task_f1
        }
        
        return metrics
    
    def train(
        self,
        n_epochs: int,
        save_dir: str,
        early_stopping_patience: int = 25,
        use_concept_intervention: bool = False,
        intervention_probability: float = 0.1
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.
        
        Args:
            n_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stopping_patience: Stop if no improvement for N epochs
            use_concept_intervention: Randomly use GT concepts during training
            intervention_probability: Probability of intervention per batch
            
        Returns:
            history: Training and validation history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_epoch = 0
        patience_counter = 0
        
        print(f"Training strategy: {self.training_strategy}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {n_epochs}\n")
        
        for epoch in range(1, n_epochs + 1):
            # Determine if using intervention this epoch
            use_intervention = (
                use_concept_intervention and
                np.random.rand() < intervention_probability
            )
            
            # Train
            train_metrics = self.train_epoch(epoch, use_intervention)
            self.train_history['concept_loss'].append(train_metrics['concept_loss'])
            self.train_history['task_loss'].append(train_metrics['task_loss'])
            self.train_history['total_loss'].append(train_metrics['total_loss'])
            
            # Evaluate
            val_metrics = self.evaluate()
            self.val_history['concept_acc'].append(val_metrics['concept_acc'])
            self.val_history['task_acc'].append(val_metrics['task_acc'])
            self.val_history['task_f1'].append(val_metrics['task_f1'])
            
            # Print progress
            print(f"Epoch {epoch}/{n_epochs}:")
            print(f"  Train - Concept Loss: {train_metrics['concept_loss']:.4f}, "
                  f"Task Loss: {train_metrics['task_loss']:.4f}")
            print(f"  Val   - Concept Acc: {val_metrics['concept_acc']:.4f}, "
                  f"Task Acc: {val_metrics['task_acc']:.4f}, "
                  f"Task F1: {val_metrics['task_f1']:.4f}")
            
            # Save best model (use >= to save first epoch even if F1=0)
            if val_metrics['task_f1'] >= self.best_val_acc:
                self.best_val_acc = val_metrics['task_f1']
                best_epoch = epoch
                patience_counter = 0
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                self.model.save(checkpoint_path)
                print(f"  ✓ New best model saved (F1: {self.best_val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best epoch: {best_epoch} (F1: {self.best_val_acc:.4f})")
                break
            
            print()
        
        # Final summary
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best validation F1: {self.best_val_acc:.4f} (epoch {best_epoch})")
        print(f"Final concept accuracy: {self.val_history['concept_acc'][-1]:.4f}")
        print(f"Final task accuracy: {self.val_history['task_acc'][-1]:.4f}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_f1': self.best_val_acc,
            'best_epoch': best_epoch
        }


def train_cbm_sequential(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    concept_epochs: int = 20,
    task_epochs: int = 10,
    save_dir: str = './outputs'
) -> Dict[str, any]:
    """
    Train CBM with sequential strategy:
    1. Train concept encoder
    2. Freeze concepts, train task predictor
    
    Args:
        model: CBM model
        train_loader: Training data
        val_loader: Validation data
        device: Device
        concept_epochs: Epochs for concept training
        task_epochs: Epochs for task predictor training
        save_dir: Save directory
        
    Returns:
        training_history: Complete training history
    """
    print("=" * 70)
    print("SEQUENTIAL TRAINING: STAGE 1 - CONCEPT ENCODER")
    print("=" * 70)
    
    # Stage 1: Train concepts
    trainer = CBMTrainer(
        model, train_loader, val_loader,
        device=device,
        concept_loss_weight=1.0,
        training_strategy='independent'
    )
    
    history_stage1 = trainer.train(
        n_epochs=concept_epochs,
        save_dir=os.path.join(save_dir, 'stage1')
    )
    
    # Stage 2: Freeze concepts, train task predictor
    print("\n" + "=" * 70)
    print("SEQUENTIAL TRAINING: STAGE 2 - TASK PREDICTOR")
    print("=" * 70)
    
    model.freeze_concepts()
    
    trainer2 = CBMTrainer(
        model, train_loader, val_loader,
        device=device,
        concept_loss_weight=0.0,  # Only task loss
        training_strategy='independent'
    )
    
    history_stage2 = trainer2.train(
        n_epochs=task_epochs,
        save_dir=os.path.join(save_dir, 'stage2')
    )
    
    return {
        'stage1': history_stage1,
        'stage2': history_stage2
    }
