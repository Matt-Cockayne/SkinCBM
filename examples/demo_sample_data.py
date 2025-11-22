"""
Demo script to visualize CBM predictions on sample dermoscopy cases.
Shows the model's concept predictions and final diagnosis with ground truth comparison.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.basic_cbm import ConceptBottleneckModel
from src.utils.visualization import plot_prediction_with_concepts


def preprocess_image(image_path):
    """Load and preprocess image for model input."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Also return unnormalized image for visualization
    viz_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_viz = viz_transform(Image.open(image_path).convert('RGB'))
    image_viz = image_viz.permute(1, 2, 0).numpy()
    
    return image_tensor, image_viz


def load_sample_cases():
    """Load sample dermoscopy cases with ground truth."""
    # Define sample cases with their metadata
    # Concept order: pigment_network, blue_whitish_veil, vascular_structures, streaks, pigmentation, dots_and_globules, regression
    # Concept mapping varies by concept - see derm7pt dataset for numeric encoding
    # All cases are from TEST set
    cases = [
        {
            'case_num': 7,
            'image': 'notebooks/sample_data_derm7pt/case_007_dermoscopic.jpg',
            'diagnosis': 'Non-Melanoma',  # basal cell carcinoma
            'concepts': [0, 0, 1, 0, 0, 0, 0]  # PN:absent, BWV:absent, VS:arborizing=1, Streaks:absent, Pig:absent, DaG:absent, Reg:absent
        },
        {
            'case_num': 596,
            'image': 'notebooks/sample_data_derm7pt/case_596_dermoscopic.jpg',
            'diagnosis': 'Melanoma',  # melanoma (in situ), 7pt=3, low complexity
            'concepts': [2, 0, 0, 0, 1, 2, 0]  # PN:atypical, BWV:absent, VS:absent, Streaks:absent, Pig:diffuse_regular=1, DaG:irregular, Reg:absent
        },
        {
            'case_num': 578,
            'image': 'notebooks/sample_data_derm7pt/case_578_dermoscopic.jpg',
            'diagnosis': 'Melanoma',  # melanoma (in situ), 7pt=7, high complexity
            'concepts': [2, 1, 0, 2, 2, 2, 0]  # PN:atypical, BWV:present, VS:absent, Streaks:irregular, Pig:localized_irregular=2, DaG:irregular, Reg:absent
        },
        {
            'case_num': 657,
            'image': 'notebooks/sample_data_derm7pt/case_657_dermoscopic.jpg',
            'diagnosis': 'Melanoma',  # melanoma (< 0.76mm), 7pt=8, very high complexity
            'concepts': [2, 1, 0, 2, 2, 2, 1]  # PN:atypical, BWV:present, VS:absent, Streaks:irregular, Pig:diffuse_irregular=2, DaG:irregular, Reg:blue_areas=1
        }
    ]
    
    return cases


def run_demo(model_path):
    """Run inference demo on sample cases with improved visualizations."""
    
    print("="*70)
    print("CBM Dermoscopy Demo - Enhanced Visualizations")
    print("="*70)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = ConceptBottleneckModel.load(model_path)
    model.eval()
    
    # Load sample cases
    cases = load_sample_cases()
    print(f"Loaded {len(cases)} sample cases\n")
    
    # Concept names
    concept_names = [
        "Pigment Network",
        "Blue-Whitish Veil",
        "Vascular Structures",
        "Streaks",
        "Pigmentation",
        "Dots & Globules",
        "Regression"
    ]
    
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each case
    for case in cases:
        print(f"\n{'='*70}")
        print(f"Processing Case {case['case_num']}: {case['diagnosis']}")
        print(f"{'='*70}")
        
        # Load and preprocess image
        if not os.path.exists(case['image']):
            print(f"Warning: Image not found: {case['image']}")
            continue
        
        image_tensor, image_viz = preprocess_image(case['image'])
        
        # Run inference
        with torch.no_grad():
            concept_preds, task_logits = model(image_tensor)
        
        # Process concept predictions
        # Reshape from [1, 21] to [1, 7, 3] for 7 concepts × 3 classes
        concept_preds_reshaped = concept_preds.reshape(1, 7, 3)
        concept_pred_classes = concept_preds_reshaped.argmax(dim=2).squeeze().cpu().numpy()
        
        # Process task prediction
        task_probs = torch.softmax(task_logits, dim=1)
        melanoma_prob = task_probs[0, 1].item()
        
        # Print results
        class_labels = ['Absent', 'Regular', 'Irregular']
        print("\nConcept Predictions:")
        for i, name in enumerate(concept_names):
            pred_class = int(concept_pred_classes[i])
            gt_class = case['concepts'][i]
            pred_label = class_labels[pred_class]
            gt_label = class_labels[gt_class]
            match = '✓' if pred_class == gt_class else '✗'
            print(f"  {name:.<30} Pred: {pred_label:<10} GT: {gt_label:<10} {match}")
        
        diagnosis_pred = "Melanoma" if melanoma_prob > 0.5 else "Non-Melanoma"
        print(f"\nDiagnosis Prediction: {diagnosis_pred} ({melanoma_prob:.1%})")
        print(f"Ground Truth: {case['diagnosis']}")
        
        # Create visualization
        task_pred_str = f"{diagnosis_pred} ({melanoma_prob:.3f})"
        
        fig = plot_prediction_with_concepts(
            image=image_viz,
            concept_preds=concept_pred_classes,
            concept_gts=np.array(case['concepts']),
            concept_names=concept_names,
            task_pred=task_pred_str,
            task_gt=case['diagnosis']
        )
        
        # Save figure
        output_path = os.path.join(output_dir, f'demo_case_{case["case_num"]}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n✓ Visualization saved: {output_path}")
    
    print("\n" + "="*70)
    print("Demo complete! Enhanced visualizations saved to outputs/")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CBM demo with enhanced visualizations')
    parser.add_argument('--model_path', type=str, 
                       default='trained_models/derm7pt_best/best_model.pth',
                       help='Path to trained model checkpoint')
    
    args = parser.parse_args()
    
    run_demo(args.model_path)
