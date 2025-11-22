"""
Train a basic Concept Bottleneck Model on skin cancer data.

This script demonstrates the simplest way to train a CBM:
- Load data
- Create model
- Train with joint strategy
- Evaluate and save

Example usage:
    python examples/train_basic_cbm.py \
        --dataset derm7pt \
        --data_path ./data \
        --epochs 50 \
        --output_dir ./outputs/basic_cbm
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.basic_cbm import ConceptBottleneckModel
from src.data.derm7pt_adapter import create_derm7pt_dataloaders
from src.training.trainer import CBMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a basic CBM')
    
    # Data args
    parser.add_argument('--dataset', type=str, default='derm7pt',
                       choices=['derm7pt', 'skincon'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data directory')
    
    # Model args
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='Backbone architecture (resnet50, efficientnet_b0, etc.)')
    parser.add_argument('--task_architecture', type=str, default='linear',
                        help='Task predictor architecture (only "linear" is implemented)')    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--concept_loss_weight', type=float, default=1.0,
                       help='Weight for concept loss vs task loss')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='./outputs/basic_cbm',
                       help='Directory to save outputs')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    if args.dataset == 'derm7pt':
        train_loader, val_loader, test_loader = create_derm7pt_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            image_size=(224, 224)
        )
        num_concepts = 7
        num_classes = 2
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet implemented")
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create model
    print("Creating model...")
    model = ConceptBottleneckModel(
        num_concepts=num_concepts,
        num_classes=num_classes,
        num_classes_per_concept=3,  # Derm7pt has 3-class concepts (absent/regular/irregular)
        backbone=args.backbone,
        pretrained=True
    )
    
    print(f"Model: {args.backbone} backbone + linear task predictor")
    print(f"Pretrained weights: ImageNet (loaded via timm)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    print("Starting training...")
    trainer = CBMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        concept_loss_weight=args.concept_loss_weight,
        training_strategy='joint'
    )
    
    history = trainer.train(
        n_epochs=args.epochs,
        save_dir=args.output_dir,
        early_stopping_patience=25
    )
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print("\nTest Set Results:")
    print(f"  Concept Accuracy: {test_metrics['concept_acc']:.4f}")
    print(f"  Task Accuracy:    {test_metrics['task_acc']:.4f}")
    print(f"  Task F1 Score:    {test_metrics['task_f1']:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.backbone} + linear predictor\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Concept Accuracy: {test_metrics['concept_acc']:.4f}\n")
        f.write(f"  Task Accuracy:    {test_metrics['task_acc']:.4f}\n")
        f.write(f"  Task F1 Score:    {test_metrics['task_f1']:.4f}\n")
    
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "=" * 70)
    print("Training complete! 🎉")
    print("=" * 70)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  - best_model.pth: Best model checkpoint")
    print(f"  - results.txt: Test set metrics")


if __name__ == '__main__':
    main()
