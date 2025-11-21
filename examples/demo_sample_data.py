"""
Demo script using sample derm7pt data for quick testing.

This demonstrates CBM inference on the 3 sample cases without needing
the full dataset.
"""

import json
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.basic_cbm import ConceptBottleneckModel


def load_sample_data(sample_dir='notebooks/sample_data_derm7pt'):
    """Load the sample cases from JSON metadata."""
    metadata_path = os.path.join(sample_dir, 'cases_metadata.json')
    
    with open(metadata_path, 'r') as f:
        cases = json.load(f)
    
    return cases


def preprocess_image(image_path):
    """Preprocess image for CBM inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


def extract_concepts_from_metadata(case):
    """
    Extract concept values from case metadata.
    
    Maps the 7-point checklist features to their numeric representations.
    Returns the numeric label for each category (not binary).
    """
    features = case['clinical_features']
    
    # Map to numeric values based on dataset.py definitions
    # pigment_network: 0=absent, 1=typical, 2=atypical
    pn = 2 if features['pigment_network'] == 'atypical' else (1 if features['pigment_network'] == 'typical' else 0)
    
    # blue_whitish_veil: 0=absent, 1=present
    bwv = 1 if features['blue_whitish_veil'] == 'present' else 0
    
    # vascular_structures: 0=absent, 1=regular types, 2=irregular types
    vs_irregular = ['dotted', 'linear irregular']
    vs = 2 if features['vascular_structures'] in vs_irregular else (0 if features['vascular_structures'] == 'absent' else 1)
    
    # streaks: 0=absent, 1=regular, 2=irregular
    str_val = 2 if features['streaks'] == 'irregular' else (1 if features['streaks'] == 'regular' else 0)
    
    # pigmentation: 0=absent, 1=regular, 2=irregular
    pig = 2 if 'irregular' in features['pigmentation'].lower() else (0 if features['pigmentation'] == 'absent' else 1)
    
    # dots_and_globules: 0=absent, 1=regular, 2=irregular
    dag = 2 if features['dots_and_globules'] == 'irregular' else (1 if features['dots_and_globules'] == 'regular' else 0)
    
    # regression_structures: 0=absent, 1=present
    reg = 0 if features['regression'] == 'absent' else 1
    
    concepts = [pn, bwv, vs, str_val, pig, dag, reg]
    
    return torch.tensor(concepts, dtype=torch.float32).unsqueeze(0)


def visualize_prediction(case, concepts, prediction, sample_dir='notebooks/sample_data_derm7pt'):
    """Visualize the case with predictions."""
    concept_names = [
        "Pigment Network",
        "Blue-Whitish Veil",
        "Vascular Structures",
        "Streaks",
        "Pigmentation",
        "Dots & Globules",
        "Regression Structures"
    ]
    
    # Load images
    derm_path = os.path.join(sample_dir, case['dermoscopic_img'])
    clinic_path = os.path.join(sample_dir, case['clinical_img'])
    
    derm_img = Image.open(derm_path)
    clinic_img = Image.open(clinic_path)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show images
    axes[0].imshow(clinic_img)
    axes[0].set_title(f"Clinical Image\nCase {case['case_num']}")
    axes[0].axis('off')
    
    axes[1].imshow(derm_img)
    axes[1].set_title(f"Dermoscopic Image\nDiagnosis: {case['diagnosis']}")
    axes[1].axis('off')
    
    # Show concept predictions
    concepts_np = concepts[0].cpu().numpy()
    colors = ['green' if c > 0.5 else 'lightgray' for c in concepts_np]
    
    axes[2].barh(range(7), concepts_np, color=colors)
    axes[2].set_yticks(range(7))
    axes[2].set_yticklabels(concept_names, fontsize=9)
    axes[2].set_xlabel('Concept Value')
    axes[2].set_title(f'Predicted Concepts\nDiagnosis: {"Malignant" if prediction > 0.5 else "Benign"}')
    axes[2].set_xlim([0, max(3, concepts_np.max() + 0.5)])  # Scale to accommodate multi-class
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def demo_inference(model_path=None):
    """Run inference demo on sample cases."""
    
    # Load sample cases
    cases = load_sample_data()
    print(f"Loaded {len(cases)} sample cases\n")
    
    # Initialize model (or load pretrained if available)
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = ConceptBottleneckModel.load(model_path)
    else:
        print("Initializing untrained model for demo")
        model = ConceptBottleneckModel(
            num_concepts=7,
            num_classes=2,
            backbone='resnet50',
            pretrained=True
        )
    
    model.eval()
    
    # Process each case
    for case in cases:
        print(f"\n{'='*60}")
        print(f"Case {case['case_num']}: {case['diagnosis']}")
        print(f"{'='*60}")
        
        # Load and preprocess image
        derm_path = os.path.join('notebooks/sample_data_derm7pt', case['dermoscopic_img'])
        image = preprocess_image(derm_path)
        
        # Get ground truth concepts from metadata
        gt_concepts = extract_concepts_from_metadata(case)
        
        # Run inference
        with torch.no_grad():
            pred_concepts, pred_logits = model(image)
        
        pred_label = torch.softmax(pred_logits, dim=1)[0, 1].item()
        
        # Display results
        print("\nGround Truth Concepts:")
        concept_names = [
            "Pigment Network",
            "Blue-Whitish Veil",
            "Vascular Structures",
            "Streaks",
            "Pigmentation",
            "Dots & Globules",
            "Regression"
        ]
        
        for name, gt_val, pred_val in zip(concept_names, 
                                           gt_concepts[0].tolist(), 
                                           pred_concepts[0].tolist()):
            print(f"  {name:.<35} GT: {gt_val:.0f}  Pred: {pred_val:.3f}")
        
        print(f"\nPredicted Diagnosis: {'Malignant' if pred_label > 0.5 else 'Benign'} ({pred_label:.3f})")
        print(f"Ground Truth: {case['diagnosis']}")
        
        # Visualize
        fig = visualize_prediction(case, pred_concepts, pred_label)
        plt.savefig(f"outputs/demo_case_{case['case_num']}.png", dpi=150, bbox_inches='tight')
        print(f"Saved visualization to outputs/demo_case_{case['case_num']}.png")
        plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CBM demo on sample derm7pt data')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Run demo
    demo_inference(args.model_path)
    
    print("\n" + "="*60)
    print("Demo complete! Check outputs/ for visualizations.")
    print("="*60)
