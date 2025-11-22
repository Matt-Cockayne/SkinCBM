"""
Visualization Utilities for CBM Analysis

Provides tools for visualizing:
- Concept predictions
- Activation maps
- Intervention results
- Model predictions with concept explanations
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import warnings


def plot_concept_predictions(
    concepts: np.ndarray,
    concept_names: List[str],
    ground_truth: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot predicted concept values vs ground truth.
    
    Args:
        concepts: Predicted concept values, shape [n_samples, n_concepts]
        concept_names: List of concept names
        ground_truth: Ground truth concept values (optional)
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_concepts = len(concept_names)
    x = np.arange(n_concepts)
    
    # Plot predicted values
    mean_pred = concepts.mean(axis=0)
    std_pred = concepts.std(axis=0)
    
    ax.bar(x, mean_pred, yerr=std_pred, alpha=0.7, label='Predicted', capsize=5)
    
    # Plot ground truth if available
    if ground_truth is not None:
        mean_gt = ground_truth.mean(axis=0)
        ax.scatter(x, mean_gt, color='red', s=100, label='Ground Truth', zorder=10)
    
    ax.set_xlabel('Concept', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Concept Predictions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(concept_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_intervention_comparison(
    original_concepts: np.ndarray,
    intervened_concepts: np.ndarray,
    concept_names: List[str],
    original_pred: str,
    intervened_pred: str,
    case_info: str = None,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Visualize the effect of concept intervention with clear before/after comparison.
    
    Args:
        original_concepts: Original concept class values [num_concepts]
        intervened_concepts: Concept values after intervention [num_concepts]
        concept_names: List of concept names
        original_pred: Original prediction label with confidence
        intervened_pred: Prediction after intervention with confidence
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], height_ratios=[1, 1], 
                          hspace=0.3, wspace=0.3)
    
    # Parse confidence values
    import re
    orig_conf = float(re.findall(r'\(([\d.]+)\)', original_pred)[0]) if '(' in original_pred else 0.5
    interv_conf = float(re.findall(r'\(([\d.]+)\)', intervened_pred)[0]) if '(' in intervened_pred else 0.5
    
    n_concepts = len(concept_names)
    class_labels = ['Absent', 'Regular', 'Irregular']
    colors_map = {0: 'lightgray', 1: 'gold', 2: 'orangered'}
    
    # Top row: Concept visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Before intervention
    y_pos = np.arange(n_concepts)
    colors_before = [colors_map[int(c)] for c in original_concepts]
    bars1 = ax1.barh(y_pos, np.ones(n_concepts), color=colors_before, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add class labels on bars
    for i, (bar, val) in enumerate(zip(bars1, original_concepts)):
        ax1.text(0.5, bar.get_y() + bar.get_height()/2, class_labels[int(val)],
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(concept_names, fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_xticks([])
    ax1.set_title('Before Intervention', fontsize=13, fontweight='bold', pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    
    # After intervention - highlight changes
    colors_after = []
    for i in range(n_concepts):
        if original_concepts[i] != intervened_concepts[i]:
            colors_after.append('limegreen')  # Changed concepts in bright green
        else:
            colors_after.append(colors_map[int(intervened_concepts[i])])
    
    bars2 = ax2.barh(y_pos, np.ones(n_concepts), color=colors_after, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add class labels on bars
    for i, (bar, val) in enumerate(zip(bars2, intervened_concepts)):
        ax2.text(0.5, bar.get_y() + bar.get_height()/2, class_labels[int(val)],
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([''] * n_concepts)  # No labels on right side
    ax2.set_xlim([0, 1])
    ax2.set_xticks([])
    ax2.set_title('After Intervention', fontsize=13, fontweight='bold', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    
    # Legend for concepts
    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis('off')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', alpha=0.8, edgecolor='black', label='Absent'),
        Patch(facecolor='gold', alpha=0.8, edgecolor='black', label='Regular'),
        Patch(facecolor='orangered', alpha=0.8, edgecolor='black', label='Irregular'),
        Patch(facecolor='limegreen', alpha=0.8, edgecolor='black', label='Intervened')
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=11, frameon=True, 
                     title='Concept Classes', title_fontsize=12)
    
    # Bottom row: Prediction comparison with gauge-style visualization
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create gauge visualization
    categories = ['Original\nPrediction', 'After\nIntervention']
    confidences = [orig_conf, interv_conf]
    
    bar_width = 0.6
    x_pos = np.arange(len(categories))
    
    # Color bars by non-melanoma (blue) or melanoma (red)
    bar_colors = []
    for conf in confidences:
        if conf > 0.5:
            # Melanoma - red gradient
            intensity = (conf - 0.5) * 2  # 0 to 1
            bar_colors.append((1, 0.3 * (1-intensity), 0.3 * (1-intensity)))
        else:
            # Non-melanoma - blue gradient
            intensity = (0.5 - conf) * 2  # 0 to 1
            bar_colors.append((0.3 * (1-intensity), 0.5 * (1-intensity), 1))
    
    bars = ax3.bar(x_pos, confidences, bar_width, color=bar_colors, alpha=0.9, 
                   edgecolor='black', linewidth=2)
    
    # Add confidence values on top of bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        label = 'Melanoma' if conf > 0.5 else 'Non-Melanoma'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{label}\n{conf:.1%}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Add horizontal line at 0.5 threshold
    ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Decision Threshold')
    
    # Calculate and show change
    change = interv_conf - orig_conf
    
    # Show change as text annotation without arrow
    if abs(change) > 0.01:
        ax3.text(0.5, 0.85, f'Change: {change:+.1%}', ha='center', va='center',
                fontsize=15, fontweight='bold', 
                color='green' if change > 0 else 'red',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='black', linewidth=2.5))
    
    ax3.set_ylabel('Melanoma Probability', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax3.set_title('Diagnosis Prediction', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add case info to title if provided
    if case_info:
        plt.suptitle(f'Concept Intervention Analysis - {case_info}', fontsize=16, fontweight='bold', y=0.98)
    else:
        plt.suptitle('Concept Intervention Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def plot_concept_heatmap(
    concepts: np.ndarray,
    concept_names: List[str],
    sample_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot heatmap of concept activations across samples.
    
    Args:
        concepts: Concept values, shape [n_samples, n_concepts]
        concept_names: List of concept names
        sample_labels: Optional labels for samples
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        concepts.T,
        xticklabels=sample_labels if sample_labels else False,
        yticklabels=concept_names,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Concept Probability'},
        ax=ax
    )
    
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Concept', fontsize=12)
    ax.set_title('Concept Activation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_prediction_with_concepts(
    image: np.ndarray,
    concept_preds: np.ndarray,
    concept_gts: Optional[np.ndarray],
    concept_names: List[str],
    task_pred: str,
    task_gt: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 8)
) -> plt.Figure:
    """
    Create a comprehensive visualization showing image, concepts, and diagnosis.
    IMPROVED VERSION with better hierarchy, emphasis, and clinical design.
    
    Design Principles:
    - Visual hierarchy: Most important info (diagnosis) most prominent
    - Color psychology: Red=danger/melanoma, Blue=safe/non-melanoma, Green=correct
    - Clinical context: Professional medical color scheme
    - Threshold emphasis: Clear 50% decision boundary visualization
    - Concept sorting: Order by correctness for quick assessment
    
    Args:
        image: Input image [H, W, C]
        concept_preds: Predicted concept classes [num_concepts]
        concept_gts: Ground truth concept classes [num_concepts] (optional)
        concept_names: List of concept names
        task_pred: Task prediction string with confidence
        task_gt: Ground truth task label (optional)
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.2, 1.6, 1.6, 0.6], height_ratios=[0.4, 1.2, 1],
                          hspace=0.35, wspace=0.6, left=0.04, right=0.98, top=0.96, bottom=0.05)
    
    # Parse task prediction
    import re
    conf_match = re.findall(r'\(([\d.]+)\)', task_pred)
    confidence = float(conf_match[0]) if conf_match else 0.5
    is_melanoma = confidence > 0.5
    pred_label = 'MELANOMA' if is_melanoma else 'NON-MELANOMA'
    
    # Determine correctness
    is_correct = None
    if task_gt is not None:
        is_correct = (pred_label == task_gt.upper())
    
    # Sort concepts by correctness (incorrect first for attention)
    concept_order = list(range(len(concept_names)))
    if concept_gts is not None:
        matches = [int(concept_preds[i]) == int(concept_gts[i]) for i in range(len(concept_names))]
        concept_order = sorted(range(len(concept_names)), key=lambda i: (matches[i], i))
    
    # === TOP: DIAGNOSIS SUMMARY (spans all columns) ===
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')
    
    # Create summary box with diagnosis
    summary_color = '#4caf50' if is_correct else '#d32f2f' if is_correct is False else '#757575'
    summary_text = f"DIAGNOSIS: {pred_label}   •   Confidence: {confidence:.1%}"
    if task_gt is not None:
        match_icon = '✓' if is_correct else '✗'
        summary_text = f"{match_icon} DIAGNOSIS: {pred_label}   •   Confidence: {confidence:.1%}   •   Ground Truth: {task_gt.upper()}"
    
    ax_summary.text(0.5, 0.5, summary_text, 
                   ha='center', va='center', fontsize=18, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=1.0', facecolor=summary_color, 
                           edgecolor='black', linewidth=3))
    
    # === LEFT: DERMOSCOPIC IMAGE ===
    ax_img = fig.add_subplot(gs[1:, 0])
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title('Dermoscopic Image', fontsize=14, fontweight='bold', pad=12)
    
    # Border color based on diagnosis
    border_color = '#d32f2f' if is_melanoma else '#1976d2'
    for spine in ax_img.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)
        spine.set_visible(True)
    
    # === MIDDLE-LEFT: CONCEPT PREDICTIONS ===
    ax_concepts = fig.add_subplot(gs[1:, 1])
    
    class_labels = ['Absent', 'Regular', 'Irregular']
    colors_map = {0: '#bdbdbd', 1: '#ffc107', 2: '#ff5722'}  # Gray, Amber, Deep Orange
    
    # Reorder concepts
    ordered_names = [concept_names[i] for i in concept_order]
    ordered_preds = [concept_preds[i] for i in concept_order]
    ordered_gts = [concept_gts[i] for i in concept_order] if concept_gts is not None else None
    
    n_concepts = len(ordered_names)
    y_pos = np.arange(n_concepts)
    
    # Create horizontal bars
    colors = [colors_map[int(pred)] for pred in ordered_preds]
    bars = ax_concepts.barh(y_pos, np.ones(n_concepts) * 0.35, left=0, 
                            color=colors, alpha=0.9, edgecolor='black', linewidth=2.5)
    
    # Add prediction labels on bars
    for i, (bar, pred) in enumerate(zip(bars, ordered_preds)):
        pred_label = class_labels[int(pred)]
        ax_concepts.text(0.175, bar.get_y() + bar.get_height()/2, pred_label,
                        ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    # Add GT comparison
    if ordered_gts is not None:
        for i, (bar, pred, gt) in enumerate(zip(bars, ordered_preds, ordered_gts)):
            match = int(pred) == int(gt)
            
            # GT bar (lighter, behind)
            gt_bar = ax_concepts.barh([bar.get_y()], [0.35], left=0.38,
                                      color=colors_map[int(gt)], alpha=0.5,
                                      edgecolor='black', linewidth=1.5, linestyle='--')
            
            # GT label
            gt_label = class_labels[int(gt)]
            ax_concepts.text(0.555, bar.get_y() + bar.get_height()/2, gt_label,
                           ha='center', va='center', fontsize=10, fontweight='normal',
                           color='black', alpha=0.8)
            
            # Match indicator
            match_icon = '✓' if match else '✗'
            match_color = '#4caf50' if match else '#d32f2f'
            ax_concepts.text(0.76, bar.get_y() + bar.get_height()/2, match_icon,
                           ha='center', va='center', fontsize=16, fontweight='bold',
                           color=match_color)
    
    # Concept names on y-axis
    ax_concepts.set_yticks(y_pos)
    ax_concepts.set_yticklabels(ordered_names, fontsize=11, fontweight='bold')
    
    # Add column headers
    if concept_gts is not None:
        ax_concepts.text(0.175, n_concepts + 0.3, 'Predicted', 
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax_concepts.text(0.555, n_concepts + 0.3, 'Ground Truth',
                        ha='center', va='bottom', fontsize=12, fontweight='bold', alpha=0.8)
        ax_concepts.text(0.76, n_concepts + 0.3, 'Match',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax_concepts.set_xlim([0, 0.85])
    ax_concepts.set_ylim([-0.6, n_concepts + 0.6])
    ax_concepts.set_xticks([])
    ax_concepts.set_title('Clinical Concept Predictions', fontsize=14, fontweight='bold', pad=15)
    ax_concepts.spines['top'].set_visible(False)
    ax_concepts.spines['right'].set_visible(False)
    ax_concepts.spines['bottom'].set_visible(False)
    ax_concepts.spines['left'].set_linewidth(2)
    ax_concepts.invert_yaxis()  # Incorrect concepts at top
    
    # === MIDDLE-RIGHT: DIAGNOSIS GAUGE ===
    ax_diagnosis = fig.add_subplot(gs[1:, 2])
    
    # Horizontal probability bar with gradient
    bar_color = '#d32f2f' if is_melanoma else '#1976d2'
    bar_alpha = 0.5 + (abs(confidence - 0.5) * 1.0)  # More confident = more opaque
    
    # Background bar (full width, light gray)
    ax_diagnosis.barh([1], [1.0], height=0.5, color='#e0e0e0', alpha=0.5,
                     edgecolor='black', linewidth=2)
    
    # Probability bar
    ax_diagnosis.barh([1], [confidence], height=0.5, color=bar_color, alpha=bar_alpha,
                     edgecolor='black', linewidth=3)
    
    # Threshold line (50%)
    ax_diagnosis.axvline(x=0.5, color='black', linestyle='--', linewidth=3, alpha=0.8, zorder=10)
    ax_diagnosis.text(0.5, 1.7, 'DECISION\nTHRESHOLD', ha='center', va='bottom',
                     fontsize=10, fontweight='bold', alpha=0.8,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                             edgecolor='black', linewidth=2, alpha=0.7))
    
    # Confidence label on bar
    label_x = min(max(confidence, 0.1), 0.9)  # Keep in bounds
    ax_diagnosis.text(label_x, 1, f'{confidence:.1%}',
                     ha='center', va='center', fontsize=16, fontweight='bold',
                     color='white', bbox=dict(boxstyle='round,pad=0.5', 
                                            facecolor='black', alpha=0.85))
    
    # Region labels
    ax_diagnosis.text(0.25, 0.3, 'NON-MELANOMA', ha='center', va='center',
                     fontsize=11, fontweight='bold', color='#1976d2', alpha=0.7)
    ax_diagnosis.text(0.75, 0.3, 'MELANOMA', ha='center', va='center',
                     fontsize=11, fontweight='bold', color='#d32f2f', alpha=0.7)
    
    # GT comparison box
    if task_gt is not None:
        gt_upper = task_gt.upper()
        match_text = '✓ CORRECT DIAGNOSIS' if is_correct else '✗ INCORRECT DIAGNOSIS'
        match_color = '#4caf50' if is_correct else '#d32f2f'
        
        ax_diagnosis.text(0.5, -0.2, match_text,
                         ha='center', va='top', fontsize=13, fontweight='bold',
                         color='white',
                         bbox=dict(boxstyle='round,pad=0.6', facecolor=match_color,
                                 edgecolor='black', linewidth=3))
        ax_diagnosis.text(0.5, -0.55, f'Ground Truth: {gt_upper}',
                         ha='center', va='top', fontsize=11, fontweight='bold',
                         color='black', alpha=0.8)
    
    ax_diagnosis.set_xlim([0, 1])
    ax_diagnosis.set_ylim([-0.7, 2.0])
    ax_diagnosis.set_xlabel('Melanoma Probability', fontsize=13, fontweight='bold')
    ax_diagnosis.set_yticks([])
    ax_diagnosis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_diagnosis.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11, fontweight='bold')
    ax_diagnosis.set_title('Diagnosis Confidence', fontsize=14, fontweight='bold', pad=15)
    ax_diagnosis.spines['top'].set_visible(False)
    ax_diagnosis.spines['right'].set_visible(False)
    ax_diagnosis.spines['left'].set_visible(False)
    ax_diagnosis.spines['bottom'].set_linewidth(2)
    ax_diagnosis.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
    
    # === RIGHT: LEGEND & INFO ===
    ax_legend = fig.add_subplot(gs[1:, 3])
    ax_legend.axis('off')
    
    from matplotlib.patches import Patch
    
    # Concept classes legend
    legend_y = 0.95
    ax_legend.text(0.5, legend_y, 'CONCEPT\nCLASSES', ha='center', va='top',
                  fontsize=11, fontweight='bold', 
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                          edgecolor='black', linewidth=2))
    
    legend_elements = [
        ('Absent', '#bdbdbd'),
        ('Regular', '#ffc107'),
        ('Irregular', '#ff5722')
    ]
    
    legend_y -= 0.18
    for label, color in legend_elements:
        legend_y -= 0.11
        ax_legend.add_patch(plt.Rectangle((0.1, legend_y), 0.3, 0.08, 
                                         facecolor=color, edgecolor='black', 
                                         linewidth=2, alpha=0.9))
        ax_legend.text(0.5, legend_y + 0.04, label, ha='left', va='center',
                      fontsize=10, fontweight='bold')
    
    # Diagnosis key
    legend_y -= 0.15
    ax_legend.text(0.5, legend_y, 'DIAGNOSIS', ha='center', va='top',
                  fontsize=11, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                          edgecolor='black', linewidth=2))
    
    diagnosis_info = [
        ('< 50%', 'Non-Melanoma', '#1976d2'),
        ('≥ 50%', 'Melanoma', '#d32f2f')
    ]
    
    legend_y -= 0.18
    for threshold, label, color in diagnosis_info:
        legend_y -= 0.11
        ax_legend.add_patch(plt.Rectangle((0.1, legend_y), 0.3, 0.08,
                                         facecolor=color, edgecolor='black',
                                         linewidth=2, alpha=0.8))
        ax_legend.text(0.5, legend_y + 0.04, f'{threshold}\n{label}',
                      ha='left', va='center', fontsize=9, fontweight='bold')
    return fig


def plot_intervention_analysis(
    image: np.ndarray,
    case_num: int,
    ground_truth: str,
    interventions_data: list,
    concept_names: List[str],
    original_prob: float,
    original_concepts: np.ndarray
) -> plt.Figure:
    """
    Create intervention visualization matching the prediction plot format.
    Shows all concept interventions and highlights the best one that improves prediction.
    
    Args:
        image: Input image [H, W, C]
        case_num: Case number
        ground_truth: Ground truth diagnosis
        interventions_data: List of intervention results
        concept_names: List of concept names
        original_prob: Original melanoma probability
        original_concepts: Original concept predictions [num_concepts]
        
    Returns:
        fig: Matplotlib figure
    """
    # Find best intervention that improves prediction
    original_pred = 'MELANOMA' if original_prob > 0.5 else 'NON-MELANOMA'
    is_original_correct = (original_pred == ground_truth.upper())
    
    best_intervention = None
    best_improvement = 0
    
    for interv in interventions_data:
        if not interv['has_change']:
            continue
            
        intervened_prob = interv['intervened_prob']
        new_pred = 'MELANOMA' if intervened_prob > 0.5 else 'NON-MELANOMA'
        is_new_correct = (new_pred == ground_truth.upper())
        
        # Calculate improvement score
        if not is_original_correct and is_new_correct:
            # Fixes incorrect prediction - highest priority
            improvement = 1000 + abs(intervened_prob - original_prob)
        elif is_original_correct and is_new_correct:
            # Keeps correct prediction - check if it gets closer to correct extreme
            if ground_truth.upper() == 'MELANOMA':
                improvement = intervened_prob - original_prob if intervened_prob > original_prob else 0
            else:
                improvement = original_prob - intervened_prob if intervened_prob < original_prob else 0
        else:
            improvement = 0
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_intervention = interv
    
    # If no improving intervention, pick the one with largest absolute change
    if best_intervention is None:
        for interv in interventions_data:
            if interv['has_change']:
                change = abs(interv['intervened_prob'] - original_prob)
                if change > best_improvement:
                    best_improvement = change
                    best_intervention = interv
    
    # Setup figure matching demo format
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.2, 1.6, 1.6], height_ratios=[0.4, 1.2, 1],
                          hspace=0.35, wspace=0.5, left=0.04, right=0.98, top=0.96, bottom=0.05)
    
    # Parse diagnosis info
    is_correct = is_original_correct
    
    # Calculate best intervention diagnosis
    best_prob = best_intervention['intervened_prob'] if best_intervention else original_prob
    best_pred = 'MELANOMA' if best_prob > 0.5 else 'NON-MELANOMA'
    is_best_correct = (best_pred == ground_truth.upper())
    
    # === TOP: DIAGNOSIS SUMMARY (Two boxes side by side) ===
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')
    
    # Original diagnosis box (left side)
    summary_color_orig = '#4caf50' if is_correct else '#d32f2f'
    summary_text_orig = f"ORIGINAL DIAGNOSIS: {original_pred}   •   {original_prob:.1%}"
    if is_correct:
        summary_text_orig = f"✓ {summary_text_orig}"
    else:
        summary_text_orig = f"✗ {summary_text_orig}"
    
    ax_summary.text(0.25, 0.5, summary_text_orig, 
                   ha='center', va='center', fontsize=16, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor=summary_color_orig, 
                           edgecolor='black', linewidth=3),
                   transform=ax_summary.transAxes)
    
    # Intervened diagnosis box (right side)
    if best_intervention:
        summary_color_interv = '#4caf50' if is_best_correct else '#d32f2f'
        summary_text_interv = f"INTERVENED DIAGNOSIS: {best_pred}   •   {best_prob:.1%}"
        if is_best_correct:
            summary_text_interv = f"✓ {summary_text_interv}"
        else:
            summary_text_interv = f"✗ {summary_text_interv}"
        
        ax_summary.text(0.75, 0.5, summary_text_interv,
                       ha='center', va='center', fontsize=16, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.8', facecolor=summary_color_interv,
                               edgecolor='black', linewidth=3),
                       transform=ax_summary.transAxes)
        
        # Arrow between boxes
        ax_summary.annotate('', xy=(0.51, 0.5), xytext=(0.49, 0.5),
                           arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                           transform=ax_summary.transAxes)
    
    # Ground truth label below
    ax_summary.text(0.5, 0.1, f'Ground Truth: {ground_truth.upper()}',
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=ax_summary.transAxes)
    
    # === LEFT: DERMOSCOPIC IMAGE ===
    ax_img = fig.add_subplot(gs[1:, 0])
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title('Dermoscopic Image', fontsize=14, fontweight='bold', pad=12)
    
    border_color = '#d32f2f' if original_prob > 0.5 else '#1976d2'
    for spine in ax_img.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)
        spine.set_visible(True)
    
    # === MIDDLE-LEFT: CONCEPT INTERVENTIONS ===
    ax_concepts = fig.add_subplot(gs[1:, 1])
    
    class_labels = ['Absent', 'Regular', 'Irregular']
    colors_map = {0: '#bdbdbd', 1: '#ffc107', 2: '#ff5722'}
    
    # Sort interventions by impact
    sorted_interventions = []
    for i, interv in enumerate(interventions_data):
        if interv['has_change']:
            change = abs(interv['intervened_prob'] - original_prob)
            sorted_interventions.append((change, i, interv))
    sorted_interventions.sort(reverse=True)
    
    # Show top interventions
    n_show = min(7, len(sorted_interventions))
    y_pos = np.arange(n_show)
    
    displayed_interventions = []
    displayed_names = []
    
    for i in range(n_show):
        _, idx, interv = sorted_interventions[i]
        displayed_interventions.append(interv)
        displayed_names.append(interv['concept_name'])
    
    # Create bars showing original -> intervened
    for i, interv in enumerate(displayed_interventions):
        y = y_pos[i]
        
        concept_name = interv['concept_name']
        original_class = interv['original_class']
        new_class = interv['new_class']
        original_prob_val = interv['original_prob']
        intervened_prob_val = interv['intervened_prob']
        change = intervened_prob_val - original_prob_val
        
        is_best = (interv == best_intervention)
        
        # Special handling for "ALL CORRECTED" intervention
        if original_class == -1:  # ALL CORRECTED marker
            # Just show the text and probability change
            ax_concepts.text(0.22, y, concept_name,
                           ha='center', va='center', fontsize=11, fontweight='bold',
                           color='#ff9800',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0',
                                   edgecolor='#ff9800', linewidth=2.5))
            
            # Probability change
            arrow_color = '#d32f2f' if change > 0 else '#1976d2'
            arrow_style = '↑' if change > 0 else '↓'
            ax_concepts.text(0.58, y, f'{arrow_style} {abs(change):.1%}',
                           ha='center', va='center', fontsize=11, fontweight='bold',
                           color=arrow_color)
            
            # Best intervention marker
            if is_best:
                ax_concepts.text(0.72, y, '⭐ BEST',
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               color='#ff9800',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0',
                                       edgecolor='#ff9800', linewidth=2))
            continue
        
        # Original concept bar
        bars1 = ax_concepts.barh([y], [0.18], left=0, 
                                 color=colors_map[original_class], alpha=0.9,
                                 edgecolor='black', linewidth=2.5 if is_best else 1.5)
        
        # Label
        ax_concepts.text(0.09, y, class_labels[original_class],
                        ha='center', va='center', fontsize=10, fontweight='bold', color='black')
        
        # Arrow
        ax_concepts.annotate('', xy=(0.25, y), xytext=(0.19, y),
                            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
        
        # Intervened concept bar
        bars2 = ax_concepts.barh([y], [0.18], left=0.26,
                                 color='#76ff03', alpha=0.95,
                                 edgecolor='black', linewidth=2.5 if is_best else 1.5)
        
        # Label
        ax_concepts.text(0.35, y, class_labels[new_class],
                        ha='center', va='center', fontsize=10, fontweight='bold', color='black')
        
        # Probability change
        arrow_color = '#d32f2f' if change > 0 else '#1976d2'
        arrow_style = '↑' if change > 0 else '↓'
        ax_concepts.text(0.48, y, f'{arrow_style} {abs(change):.1%}',
                        ha='left', va='center', fontsize=10, fontweight='bold',
                        color=arrow_color)
        
        # Best intervention marker
        if is_best:
            ax_concepts.text(0.72, y, '⭐ BEST',
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='#ff9800',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0',
                                   edgecolor='#ff9800', linewidth=2))
    
    ax_concepts.set_yticks(y_pos)
    ax_concepts.set_yticklabels(displayed_names, fontsize=11, fontweight='bold')
    ax_concepts.set_xlim([0, 0.85])
    ax_concepts.set_ylim([-0.6, n_show + 0.6])
    ax_concepts.set_xticks([])
    
    # Column headers
    ax_concepts.text(0.09, n_show + 0.3, 'Original', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_concepts.text(0.35, n_show + 0.3, 'Intervened',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_concepts.text(0.58, n_show + 0.3, 'Δ Probability',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax_concepts.set_title('Concept Interventions (Sorted by Impact)', fontsize=14, fontweight='bold', pad=15)
    ax_concepts.spines['top'].set_visible(False)
    ax_concepts.spines['right'].set_visible(False)
    ax_concepts.spines['bottom'].set_visible(False)
    ax_concepts.spines['left'].set_linewidth(2)
    ax_concepts.invert_yaxis()
    
    # === MIDDLE-RIGHT: BEST INTERVENTION DIAGNOSIS ===
    ax_diagnosis = fig.add_subplot(gs[1:, 2])
    
    if best_intervention is not None:
        best_prob = best_intervention['intervened_prob']
        best_pred = 'MELANOMA' if best_prob > 0.5 else 'NON-MELANOMA'
        best_correct = (best_pred == ground_truth.upper())
        
        # Title
        ax_diagnosis.text(0.5, 1.95, 'Best Intervention Effect', ha='center', va='top',
                         fontsize=13, fontweight='bold', transform=ax_diagnosis.transData)
        
        # Intervention info box
        concept_name = best_intervention['concept_name']
        orig_class = class_labels[best_intervention['original_class']] if best_intervention['original_class'] != -1 else ''
        new_class = class_labels[best_intervention['new_class']] if best_intervention['new_class'] != -1 else ''
        
        if best_intervention['original_class'] != -1:
            info_text = f"{concept_name}\n{orig_class} → {new_class}"
        else:
            info_text = concept_name
        
        ax_diagnosis.text(0.5, 1.7, info_text, ha='center', va='top',
                         fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3e0',
                                 edgecolor='#ff9800', linewidth=2.5),
                         transform=ax_diagnosis.transData)
        
        # Probability comparison
        bar_color_orig = '#d32f2f' if original_prob > 0.5 else '#1976d2'
        bar_color_new = '#d32f2f' if best_prob > 0.5 else '#1976d2'
        
        # Background bars
        ax_diagnosis.barh([1.2], [1.0], height=0.35, color='#e0e0e0', alpha=0.5,
                         edgecolor='black', linewidth=2)
        ax_diagnosis.barh([0.6], [1.0], height=0.35, color='#e0e0e0', alpha=0.5,
                         edgecolor='black', linewidth=2)
        
        # Original probability
        bar_alpha_orig = 0.5 + (abs(original_prob - 0.5) * 1.0)
        ax_diagnosis.barh([1.2], [original_prob], height=0.35, color=bar_color_orig,
                         alpha=bar_alpha_orig, edgecolor='black', linewidth=3)
        
        # New probability
        bar_alpha_new = 0.5 + (abs(best_prob - 0.5) * 1.0)
        ax_diagnosis.barh([0.6], [best_prob], height=0.35, color=bar_color_new,
                         alpha=bar_alpha_new, edgecolor='black', linewidth=3)
        
        # Threshold lines
        ax_diagnosis.axvline(x=0.5, color='black', linestyle='--', linewidth=3, alpha=0.8, zorder=10)
        ax_diagnosis.text(0.5, 1.55, 'THRESHOLD', ha='center', va='center',
                         fontsize=9, fontweight='bold', alpha=0.8,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                 edgecolor='black', linewidth=2, alpha=0.7))
        
        # Labels
        ax_diagnosis.text(-0.02, 1.2, 'Original:', ha='right', va='center',
                         fontsize=11, fontweight='bold')
        ax_diagnosis.text(-0.02, 0.6, 'After:', ha='right', va='center',
                         fontsize=11, fontweight='bold')
        
        # Probability values on bars
        label_x_orig = min(max(original_prob, 0.1), 0.9)
        ax_diagnosis.text(label_x_orig, 1.2, f'{original_prob:.1%}',
                         ha='center', va='center', fontsize=14, fontweight='bold',
                         color='white', bbox=dict(boxstyle='round,pad=0.4',
                                                facecolor='black', alpha=0.85))
        
        label_x_new = min(max(best_prob, 0.1), 0.9)
        ax_diagnosis.text(label_x_new, 0.6, f'{best_prob:.1%}',
                         ha='center', va='center', fontsize=14, fontweight='bold',
                         color='white', bbox=dict(boxstyle='round,pad=0.4',
                                                facecolor='black', alpha=0.85))
        
        # Region labels
        ax_diagnosis.text(0.25, 0.1, 'NON-MELANOMA', ha='center', va='center',
                         fontsize=10, fontweight='bold', color='#1976d2', alpha=0.7)
        ax_diagnosis.text(0.75, 0.1, 'MELANOMA', ha='center', va='center',
                         fontsize=10, fontweight='bold', color='#d32f2f', alpha=0.7)
        
        # Result assessment
        change = best_prob - original_prob
        if not is_correct and best_correct:
            result_text = '✓ FIXES DIAGNOSIS'
            result_color = '#4caf50'
        elif is_correct and not best_correct:
            result_text = '✗ BREAKS DIAGNOSIS'
            result_color = '#f44336'
        elif (original_prob < 0.5) != (best_prob < 0.5):
            result_text = '⚡ FLIPS DIAGNOSIS'
            result_color = '#ff9800'
        else:
            if abs(change) > 0.1:
                result_text = f'{"↑" if change > 0 else "↓"} LARGE CHANGE'
                result_color = '#2196f3'
            else:
                result_text = f'{"↑" if change > 0 else "↓"} SMALL CHANGE'
                result_color = '#757575'
        
        ax_diagnosis.text(0.5, -0.2, result_text,
                         ha='center', va='top', fontsize=13, fontweight='bold',
                         color='white',
                         bbox=dict(boxstyle='round,pad=0.6', facecolor=result_color,
                                 edgecolor='black', linewidth=3))
        
        ax_diagnosis.set_xlim([-0.1, 1])
        ax_diagnosis.set_ylim([-0.4, 2.1])
    else:
        # No interventions case
        ax_diagnosis.text(0.5, 1.0, 'No Valid\nInterventions', ha='center', va='center',
                         fontsize=16, fontweight='bold', color='gray',
                         transform=ax_diagnosis.transAxes)
    
    ax_diagnosis.set_xlabel('Melanoma Probability', fontsize=13, fontweight='bold')
    ax_diagnosis.set_yticks([])
    ax_diagnosis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_diagnosis.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11, fontweight='bold')
    ax_diagnosis.set_title('Diagnosis Confidence Change', fontsize=14, fontweight='bold', pad=15)
    ax_diagnosis.spines['top'].set_visible(False)
    ax_diagnosis.spines['right'].set_visible(False)
    ax_diagnosis.spines['left'].set_visible(False)
    ax_diagnosis.spines['bottom'].set_linewidth(2)
    ax_diagnosis.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
    
    return fig


def save_figures(figures: dict, output_dir: str):
    """
    Save multiple figures to directory.
    
    Args:
        figures: Dictionary of {name: figure}
        output_dir: Output directory path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in figures.items():
        filepath = os.path.join(output_dir, f"{name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
