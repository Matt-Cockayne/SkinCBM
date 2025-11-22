"""
Concept Intervention Visualization
Matches the format of the prediction demo plots

Shows all concept interventions sorted by impact with the best intervention
highlighted and detailed in the diagnosis confidence panel.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.basic_cbm import ConceptBottleneckModel
from src.utils.visualization import plot_intervention_analysis


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
    
    viz_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_viz = viz_transform(Image.open(image_path).convert('RGB'))
    image_viz = image_viz.permute(1, 2, 0).numpy()
    
    return image_tensor, image_viz


def intervene_on_concept(model, image_tensor, concept_idx, new_class):
    """Perform intervention on a specific concept."""
    with torch.no_grad():
        original_concepts, original_logits = model(image_tensor)
        concepts_reshaped = original_concepts.reshape(1, 7, 3)
        
        intervened_concepts = concepts_reshaped.clone()
        intervened_concepts[0, concept_idx, :] = 0.0
        intervened_concepts[0, concept_idx, new_class] = 1.0
        
        intervened_concepts_flat = intervened_concepts.reshape(1, -1)
        intervened_logits = model.task_predictor(intervened_concepts_flat)
        
        original_prob = torch.softmax(original_logits, dim=1)[0, 1].item()
        intervened_prob = torch.softmax(intervened_logits, dim=1)[0, 1].item()
        original_classes = concepts_reshaped.argmax(dim=2).squeeze().cpu().numpy()
    
    return original_classes, original_prob, intervened_prob


def run_intervention_demo(model_path):
    """Run intervention demo."""
    
    print("="*70)
    print("Concept Intervention Demo - Matching Prediction Format")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = ConceptBottleneckModel.load(model_path)
    model.eval()
    
    cases = [
        {'case_num': 7, 'image': 'notebooks/sample_data_derm7pt/case_007_dermoscopic.jpg', 'diagnosis': 'Non-Melanoma'},
        {'case_num': 596, 'image': 'notebooks/sample_data_derm7pt/case_596_dermoscopic.jpg', 'diagnosis': 'Melanoma'},
        {'case_num': 578, 'image': 'notebooks/sample_data_derm7pt/case_578_dermoscopic.jpg', 'diagnosis': 'Melanoma'},
        {'case_num': 657, 'image': 'notebooks/sample_data_derm7pt/case_657_dermoscopic.jpg', 'diagnosis': 'Melanoma'}
    ]
    
    concept_names = ["Pigment Network", "Blue-Whitish Veil", "Vascular Structures",
                     "Streaks", "Pigmentation", "Dots & Globules", "Regression"]
    
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    for case in cases:
        case_num = case['case_num']
        image_path = case['image']
        ground_truth = case['diagnosis']
        
        print(f"\n{'='*70}")
        print(f"Processing Case {case_num}: {ground_truth}")
        print(f"{'='*70}")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        image_tensor, image_viz = preprocess_image(image_path)
        
        # Get original prediction
        with torch.no_grad():
            original_concepts_tensor, original_logits = model(image_tensor)
            original_concepts = original_concepts_tensor.reshape(1, 7, 3).argmax(dim=2).squeeze().cpu().numpy()
            original_prob = torch.softmax(original_logits, dim=1)[0, 1].item()
        
        # Get ground truth concepts for interventions (from test set)
        gt_concepts = {
            7: [0, 0, 1, 0, 0, 0, 0],      # Case 7: BCC, 7pt=0
            596: [2, 0, 0, 0, 1, 2, 0],    # Case 596: Melanoma in situ, 7pt=3
            578: [2, 1, 0, 2, 2, 2, 0],    # Case 578: Melanoma in situ, 7pt=7
            657: [2, 1, 0, 2, 2, 2, 1]     # Case 657: Melanoma < 0.76mm, 7pt=8
        }
        case_gt_concepts = gt_concepts.get(case_num, [0]*7)
        
        # Only test interventions for concepts that are wrongly predicted
        interventions_to_test = []
        for idx, (pred_class, gt_class, name) in enumerate(zip(original_concepts, case_gt_concepts, concept_names)):
            if pred_class != gt_class:
                # Intervene to correct value (ground truth)
                interventions_to_test.append({
                    'idx': idx,
                    'name': name,
                    'new_class': gt_class
                })
        
        interventions_data = []
        class_labels = ['Absent', 'Regular', 'Irregular']
        
        # Individual interventions (one concept at a time)
        for interv in interventions_to_test:
            concept_idx = interv['idx']
            concept_name = interv['name']
            new_class = interv['new_class']
            
            orig_classes, orig_prob, interv_prob = intervene_on_concept(
                model, image_tensor, concept_idx, new_class
            )
            
            original_class = int(orig_classes[concept_idx])
            
            interventions_data.append({
                'concept_name': concept_name,
                'concept_idx': concept_idx,
                'original_class': original_class,
                'new_class': new_class,
                'original_prob': orig_prob,
                'intervened_prob': interv_prob,
                'has_change': True
            })
            
            change = interv_prob - orig_prob
            crosses = (orig_prob < 0.5) != (interv_prob < 0.5)
            marker = "🔥" if crosses else ""
            print(f"  {marker} {concept_name:.<25} {class_labels[original_class]:>10} → {class_labels[new_class]:<10} (GT) "
                  f"| {orig_prob:.1%} → {interv_prob:.1%} ({change:+.1%})")
        
        # Add cumulative intervention (all concepts corrected)
        if len(interventions_to_test) > 0:
            print(f"\n  Correcting ALL {len(interventions_to_test)} wrong concepts simultaneously:")
            
            with torch.no_grad():
                original_concepts_tensor, original_logits = model(image_tensor)
                concepts_reshaped = original_concepts_tensor.reshape(1, 7, 3)
                
                # Correct all wrong concepts
                intervened_concepts = concepts_reshaped.clone()
                for interv in interventions_to_test:
                    concept_idx = interv['idx']
                    new_class = interv['new_class']
                    intervened_concepts[0, concept_idx, :] = 0.0
                    intervened_concepts[0, concept_idx, new_class] = 1.0
                
                intervened_concepts_flat = intervened_concepts.reshape(1, -1)
                intervened_logits = model.task_predictor(intervened_concepts_flat)
                
                cumulative_prob = torch.softmax(intervened_logits, dim=1)[0, 1].item()
            
            change = cumulative_prob - original_prob
            crosses = (original_prob < 0.5) != (cumulative_prob < 0.5)
            marker = "🔥" if crosses else ""
            
            concept_names_list = ", ".join([interv['name'] for interv in interventions_to_test])
            print(f"  {marker} ALL CONCEPTS CORRECTED........ "
                  f"| {original_prob:.1%} → {cumulative_prob:.1%} ({change:+.1%})")
            
            # Add as special intervention
            interventions_data.append({
                'concept_name': f'ALL {len(interventions_to_test)} CORRECTED',
                'concept_idx': -1,  # Special marker
                'original_class': -1,
                'new_class': -1,
                'original_prob': original_prob,
                'intervened_prob': cumulative_prob,
                'has_change': True
            })
        
        # Use new visualization function
        fig = plot_intervention_analysis(
            image=image_viz,
            case_num=case_num,
            ground_truth=ground_truth,
            interventions_data=interventions_data,
            concept_names=concept_names,
            original_prob=original_prob,
            original_concepts=original_concepts
        )
        
        if fig is not None:
            output_path = os.path.join(output_dir, f'intervention_case_{case_num}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"\n✓ Visualization saved: {output_path}")
    
    print("\n" + "="*70)
    print("Demo complete! Check outputs/intervention_case_*.png")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CBM intervention visualization')
    parser.add_argument('--model_path', type=str,
                       default='trained_models/derm7pt_best/best_model.pth',
                       help='Path to trained model checkpoint')
    
    args = parser.parse_args()
    
    run_intervention_demo(args.model_path)
