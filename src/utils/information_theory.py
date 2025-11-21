"""
Information Theory Utilities for CBM Analysis

Provides tools to compute:
- Mutual Information (MI): I(X; Y)
- Conditional MI: I(X; Y | Z)
- Synergy: Information from concept interactions
- Concept Completeness: How much information concepts capture
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import warnings


def compute_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    discrete_X: bool = True,
    discrete_Y: bool = True,
    bins: int = 10
) -> float:
    """
    Compute mutual information I(X; Y).
    
    MI measures how much knowing X reduces uncertainty about Y.
    MI = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
    
    Args:
        X: Random variable 1, shape [n_samples] or [n_samples, n_features]
        Y: Random variable 2, shape [n_samples]
        discrete_X: Whether X is discrete (otherwise continuous)
        discrete_Y: Whether Y is discrete (otherwise continuous)
        bins: Number of bins for discretizing continuous variables
        
    Returns:
        mi: Mutual information in bits
        
    Example:
        >>> concepts = model.predict_concepts(images)  # [n, 23]
        >>> labels = dataset.labels  # [n]
        >>> 
        >>> # MI between each concept and label
        >>> mi_scores = [
        >>>     compute_mutual_information(concepts[:, i], labels)
        >>>     for i in range(23)
        >>> ]
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Flatten if needed
    if X.ndim > 1 and X.shape[1] == 1:
        X = X.ravel()
    if Y.ndim > 1:
        Y = Y.ravel()
    
    # Discretize continuous variables
    if not discrete_X and X.ndim == 1:
        X = np.digitize(X, bins=np.linspace(X.min(), X.max(), bins))
    
    if not discrete_Y:
        Y = np.digitize(Y, bins=np.linspace(Y.min(), Y.max(), bins))
    
    # Compute MI
    if X.ndim == 1:
        # Single variable
        mi = mutual_info_score(X, Y)
    else:
        # Multiple variables - use sklearn's mutual_info_classif
        mi = mutual_info_classif(X.reshape(-1, 1) if X.ndim == 1 else X, Y, discrete_features=discrete_X)
        if isinstance(mi, np.ndarray):
            mi = mi[0] if len(mi) == 1 else mi.sum()
    
    # Convert to bits (sklearn returns nats)
    mi_bits = mi / np.log(2)
    
    return float(mi_bits)


def compute_concept_mi_scores(
    concepts: np.ndarray,
    labels: np.ndarray,
    concept_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute MI between each concept and the label.
    
    Identifies which individual concepts are most informative.
    
    Args:
        concepts: Concept predictions [n_samples, n_concepts]
        labels: Task labels [n_samples]
        concept_names: Optional names for concepts
        
    Returns:
        mi_scores: Dictionary mapping concept name to MI (bits)
        
    Example:
        >>> mi_scores = compute_concept_mi_scores(concepts, labels, concept_names)
        >>> print(f"Most informative: {max(mi_scores, key=mi_scores.get)}")
    """
    n_concepts = concepts.shape[1]
    
    if concept_names is None:
        concept_names = [f"concept_{i}" for i in range(n_concepts)]
    
    mi_scores = {}
    for i, name in enumerate(concept_names):
        mi = compute_mutual_information(concepts[:, i], labels, discrete_X=True, discrete_Y=True)
        mi_scores[name] = mi
    
    return mi_scores


def compute_joint_mutual_information(
    concepts: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute joint MI: I(C; Y) where C is all concepts together.
    
    This measures total information that concepts provide about the label.
    
    Args:
        concepts: All concept predictions [n_samples, n_concepts]
        labels: Task labels [n_samples]
        n_bins: Number of bins for discretization
        
    Returns:
        joint_mi: I(C; Y) in bits
    """
    # Discretize concepts into bins
    n_samples, n_concepts = concepts.shape
    
    # Create discrete representation of concept vector
    # Map each sample's concept vector to a unique bin
    concept_bins = []
    for i in range(n_concepts):
        bins = np.linspace(0, 1, n_bins + 1)
        concept_bins.append(np.digitize(concepts[:, i], bins))
    
    # Combine into single discrete variable
    # Each unique concept configuration gets a unique ID
    concept_configs = np.array(concept_bins).T
    
    # Convert to single discrete variable
    concept_ids = []
    unique_configs = {}
    for config in concept_configs:
        config_tuple = tuple(config)
        if config_tuple not in unique_configs:
            unique_configs[config_tuple] = len(unique_configs)
        concept_ids.append(unique_configs[config_tuple])
    
    concept_ids = np.array(concept_ids)
    
    # Compute MI
    joint_mi = mutual_info_score(concept_ids, labels) / np.log(2)
    
    return float(joint_mi)


def compute_synergy(
    concepts: np.ndarray,
    labels: np.ndarray,
    concept_names: Optional[List[str]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute synergy: information from concept interactions.
    
    Synergy = I(C; Y) - Σ I(c_i; Y)
    
    Where:
    - I(C; Y) is joint MI (all concepts together)
    - Σ I(c_i; Y) is sum of individual concept MIs
    
    Positive synergy means concepts interact to provide more information
    than they do individually.
    
    Args:
        concepts: Concept predictions [n_samples, n_concepts]
        labels: Task labels [n_samples]
        concept_names: Optional concept names
        
    Returns:
        synergy: Synergy in bits
        breakdown: Dictionary with 'joint_mi', 'individual_mi_sum', 'synergy'
        
    Example:
        >>> synergy, breakdown = compute_synergy(concepts, labels)
        >>> print(f"Synergy: {synergy:.3f} bits")
        >>> print(f"  Joint MI: {breakdown['joint_mi']:.3f} bits")
        >>> print(f"  Individual sum: {breakdown['individual_mi_sum']:.3f} bits")
        >>> 
        >>> if synergy > 0.1:
        >>>     print("Strong concept interactions detected!")
        >>>     print("Consider using non-linear task predictor.")
    """
    # Compute individual MIs
    individual_mi_scores = compute_concept_mi_scores(concepts, labels, concept_names)
    individual_mi_sum = sum(individual_mi_scores.values())
    
    # Compute joint MI
    joint_mi = compute_joint_mutual_information(concepts, labels)
    
    # Synergy is the difference
    synergy = joint_mi - individual_mi_sum
    
    breakdown = {
        'joint_mi': joint_mi,
        'individual_mi_sum': individual_mi_sum,
        'synergy': synergy,
        'individual_mi_scores': individual_mi_scores
    }
    
    return synergy, breakdown


def compute_concept_completeness(
    images: np.ndarray,
    concepts: np.ndarray,
    labels: np.ndarray,
    image_embedding_model: Optional[torch.nn.Module] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute concept completeness: how much information concepts capture.
    
    Completeness measures vocabulary loss:
    Vocab Loss = I(X; Y) - I(C; Y)
    
    Where:
    - I(X; Y) is max possible information (from images)
    - I(C; Y) is information captured by concepts
    
    Low vocab loss → concepts are complete (capture most information)
    High vocab loss → concepts are incomplete (missing important information)
    
    Args:
        images: Raw images or image embeddings [n_samples, ...]
        concepts: Concept predictions [n_samples, n_concepts]
        labels: Task labels [n_samples]
        image_embedding_model: Optional model to embed images
        
    Returns:
        completeness: Ratio I(C; Y) / I(X; Y)
        breakdown: Dictionary with vocab_loss, concept_mi, image_mi
        
    Example:
        >>> completeness, breakdown = compute_concept_completeness(
        >>>     image_embeddings, concepts, labels
        >>> )
        >>> print(f"Concept completeness: {completeness:.1%}")
        >>> print(f"Vocabulary loss: {breakdown['vocab_loss']:.3f} bits")
    """
    # Embed images if model provided
    if image_embedding_model is not None:
        with torch.no_grad():
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)
            images = image_embedding_model(images).cpu().numpy()
    
    # Flatten images for MI computation
    if images.ndim > 2:
        images = images.reshape(images.shape[0], -1)
    
    # Subsample features if too many (for computational efficiency)
    if images.shape[1] > 512:
        indices = np.random.choice(images.shape[1], 512, replace=False)
        images = images[:, indices]
    
    # Compute I(X; Y) - max information in images
    image_mi = compute_mutual_information(
        images, labels, discrete_X=False, discrete_Y=True, bins=20
    )
    
    # Compute I(C; Y) - information in concepts
    concept_mi = compute_joint_mutual_information(concepts, labels)
    
    # Vocabulary loss
    vocab_loss = image_mi - concept_mi
    
    # Completeness ratio
    completeness = concept_mi / image_mi if image_mi > 0 else 0.0
    
    breakdown = {
        'vocab_loss': vocab_loss,
        'concept_mi': concept_mi,
        'image_mi': image_mi,
        'completeness': completeness
    }
    
    return completeness, breakdown


def analyze_cbm_information(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    concept_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Comprehensive information-theoretic analysis of a CBM.
    
    Computes:
    - Individual concept MI scores
    - Joint MI
    - Synergy
    - Concept completeness (if possible)
    
    Args:
        model: Trained CBM model
        dataloader: DataLoader for evaluation
        device: Device to run on
        concept_names: Concept names
        
    Returns:
        analysis: Dictionary with all computed metrics
        
    Example:
        >>> analysis = analyze_cbm_information(model, test_loader, 'cuda')
        >>> 
        >>> print("Individual Concept MI:")
        >>> for name, mi in analysis['individual_mi'].items():
        >>>     print(f"  {name}: {mi:.4f} bits")
        >>> 
        >>> print(f"\nJoint MI: {analysis['joint_mi']:.4f} bits")
        >>> print(f"Synergy: {analysis['synergy']:.4f} bits")
        >>> print(f"Completeness: {analysis['completeness']:.1%}")
    """
    model.eval()
    
    all_concepts = []
    all_labels = []
    all_images = []
    
    # Collect predictions
    with torch.no_grad():
        for images, true_concepts, labels in dataloader:
            images = images.to(device)
            
            # Get concept predictions
            concepts, _ = model(images, return_concepts=True)
            
            all_concepts.append(concepts.cpu().numpy())
            all_labels.append(labels.numpy())
            all_images.append(images.cpu().numpy())
    
    # Concatenate
    all_concepts = np.vstack(all_concepts)
    all_labels = np.concatenate(all_labels)
    all_images = np.vstack([img.reshape(img.shape[0], -1) for img in all_images])
    
    # Compute metrics
    individual_mi = compute_concept_mi_scores(all_concepts, all_labels, concept_names)
    joint_mi = compute_joint_mutual_information(all_concepts, all_labels)
    synergy, synergy_breakdown = compute_synergy(all_concepts, all_labels, concept_names)
    
    # Concept completeness (using flattened images as proxy)
    completeness, completeness_breakdown = compute_concept_completeness(
        all_images, all_concepts, all_labels
    )
    
    analysis = {
        'individual_mi': individual_mi,
        'joint_mi': joint_mi,
        'synergy': synergy,
        'synergy_breakdown': synergy_breakdown,
        'completeness': completeness,
        'completeness_breakdown': completeness_breakdown,
        'n_samples': len(all_labels)
    }
    
    return analysis


def print_information_analysis(analysis: Dict[str, any]):
    """
    Pretty-print information analysis results.
    
    Args:
        analysis: Output from analyze_cbm_information
    """
    print("=" * 70)
    print("INFORMATION-THEORETIC ANALYSIS")
    print("=" * 70)
    
    print(f"\nDataset size: {analysis['n_samples']} samples\n")
    
    # Individual MI scores
    print("Individual Concept MI (I(c_i; Y)):")
    print("-" * 50)
    individual_mi = analysis['individual_mi']
    sorted_concepts = sorted(individual_mi.items(), key=lambda x: x[1], reverse=True)
    
    for name, mi in sorted_concepts[:10]:  # Top 10
        bar = "█" * int(mi * 50)
        print(f"  {name:30s} {mi:6.4f} bits {bar}")
    
    if len(sorted_concepts) > 10:
        print(f"  ... ({len(sorted_concepts) - 10} more concepts)")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nJoint MI (I(C; Y)):              {analysis['joint_mi']:.4f} bits")
    print(f"Sum of Individual MI:             {analysis['synergy_breakdown']['individual_mi_sum']:.4f} bits")
    print(f"Synergy (interactions):           {analysis['synergy']:.4f} bits")
    if analysis['joint_mi'] > 0:
        print(f"  → {(analysis['synergy'] / analysis['joint_mi'] * 100):.1f}% of information from synergy")
    else:
        print(f"  → N/A (zero joint MI)")
    
    if 'completeness_breakdown' in analysis:
        cb = analysis['completeness_breakdown']
        print(f"\nImage MI (I(X; Y)):               {cb['image_mi']:.4f} bits")
        print(f"Concept MI (I(C; Y)):             {cb['concept_mi']:.4f} bits")
        print(f"Vocabulary Loss:                  {cb['vocab_loss']:.4f} bits")
        print(f"Concept Completeness:             {cb['completeness']:.1%}")
    
    print("\n" + "=" * 70)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if analysis['synergy'] > 0.1:
        print("  ✓ High synergy detected! Concepts interact strongly.")
        print("    → Recommendation: Consider non-linear task predictor for production use")
    else:
        print("  • Low synergy. Concepts mostly independent.")
        print("    → Recommendation: Linear task predictor is sufficient")
    
    if 'completeness' in analysis:
        if analysis['completeness'] > 0.85:
            print("  ✓ High completeness! Concepts capture most information.")
        elif analysis['completeness'] > 0.7:
            print("  • Moderate completeness. Some information missing.")
            print("    → Consider: Adding more concepts or refining definitions")
        else:
            print("  ⚠ Low completeness! Significant vocabulary loss.")
            print("    → Action needed: Add important missing concepts")
    
    print("=" * 70)
