"""
Concept Intervention Analysis for Test Set
Generates tables showing predictions and intervention effects across the test dataset.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class InterventionAnalyzer:
    """Analyzes concept interventions across a test dataset."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the intervention analyzer.
        
        Args:
            model: Trained ConceptBottleneckModel
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.concept_names = [
            "Pigment Network", "Blue-Whitish Veil", "Vascular Structures",
            "Streaks", "Pigmentation", "Dots & Globules", "Regression"
        ]
        self.class_labels = ['Absent', 'Regular', 'Irregular']
    
    def predict(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model prediction.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            concept_preds: Concept class predictions [B, 7]
            concept_probs: Concept probabilities [B, 7, 3]
            task_probs: Task probabilities [B, 2]
        """
        with torch.no_grad():
            images = images.to(self.device)
            concepts, task_logits = self.model(images)
            
            # Reshape concepts: [B, 21] -> [B, 7, 3]
            concept_probs = concepts.reshape(-1, 7, 3)
            concept_preds = concept_probs.argmax(dim=2).cpu().numpy()
            
            # Task predictions
            task_probs = torch.softmax(task_logits, dim=1).cpu().numpy()
            
        return concept_preds, concept_probs.cpu().numpy(), task_probs
    
    def intervene_single_concept(
        self, 
        images: torch.Tensor,
        concept_idx: int,
        new_class: int
    ) -> np.ndarray:
        """
        Perform intervention on a single concept for a batch.
        
        Args:
            images: Batch of images [B, C, H, W]
            concept_idx: Index of concept to intervene on (0-6)
            new_class: New class to set (0=Absent, 1=Regular, 2=Irregular)
            
        Returns:
            intervened_probs: Task probabilities after intervention [B, 2]
        """
        with torch.no_grad():
            images = images.to(self.device)
            concepts, _ = self.model(images)
            
            # Reshape and intervene
            concepts_reshaped = concepts.reshape(-1, 7, 3)
            intervened_concepts = concepts_reshaped.clone()
            intervened_concepts[:, concept_idx, :] = 0.0
            intervened_concepts[:, concept_idx, int(new_class)] = 1.0
            
            # Get new predictions
            intervened_concepts_flat = intervened_concepts.reshape(-1, 21)
            intervened_logits = self.model.task_predictor(intervened_concepts_flat)
            intervened_probs = torch.softmax(intervened_logits, dim=1).cpu().numpy()
            
        return intervened_probs
    
    def analyze_test_set(
        self,
        test_loader,
        output_dir: str = 'outputs/intervention_analysis'
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze interventions across entire test set.
        
        Args:
            test_loader: DataLoader for test set
            output_dir: Directory to save results
            
        Returns:
            results: Dictionary of result DataFrames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for results
        all_results = []
        
        print("Analyzing test set interventions...")
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # Derm7ptCBMAdapter returns (image, concepts, label)
            images, concept_labels, labels = batch
            labels = labels.numpy() if torch.is_tensor(labels) else labels
            concept_labels = concept_labels.numpy() if torch.is_tensor(concept_labels) else concept_labels
            
            # Get original predictions
            concept_preds, concept_probs, task_probs = self.predict(images)
            original_melanoma_prob = task_probs[:, 1]
            original_pred = (original_melanoma_prob > 0.5).astype(int)
            
            batch_size = images.size(0)
            
            # For each sample in batch
            for i in range(batch_size):
                sample_id = batch_idx * test_loader.batch_size + i
                
                # Base record
                record = {
                    'sample_id': sample_id,
                    'true_label': labels[i],
                    'original_pred': original_pred[i],
                    'original_prob': original_melanoma_prob[i],
                    'correct': (original_pred[i] == labels[i])
                }
                
                # Add concept predictions
                for j, name in enumerate(self.concept_names):
                    record[f'{name}_pred'] = concept_preds[i, j]
                    record[f'{name}_true'] = concept_labels[i, j]
                    record[f'{name}_correct'] = (concept_preds[i, j] == concept_labels[i, j])
                
                # Intervene on each wrong concept
                for j, name in enumerate(self.concept_names):
                    if concept_preds[i, j] != concept_labels[i, j]:
                        # Intervene to ground truth
                        gt_class = int(concept_labels[i, j])
                        intervened_probs = self.intervene_single_concept(
                            images[i:i+1], j, gt_class
                        )
                        intervened_prob = intervened_probs[0, 1]
                        intervened_pred = int(intervened_prob > 0.5)
                        
                        record[f'{name}_intervention_prob'] = intervened_prob
                        record[f'{name}_intervention_pred'] = intervened_pred
                        record[f'{name}_intervention_change'] = intervened_prob - original_melanoma_prob[i]
                        record[f'{name}_intervention_fixes'] = (
                            (original_pred[i] != labels[i]) and 
                            (intervened_pred == labels[i])
                        )
                    else:
                        record[f'{name}_intervention_prob'] = np.nan
                        record[f'{name}_intervention_pred'] = np.nan
                        record[f'{name}_intervention_change'] = np.nan
                        record[f'{name}_intervention_fixes'] = False
                
                # Cumulative intervention (all wrong concepts)
                wrong_concepts = [j for j in range(7) if concept_preds[i, j] != concept_labels[i, j]]
                if len(wrong_concepts) > 0:
                    # Intervene on all wrong concepts at once
                    with torch.no_grad():
                        img = images[i:i+1].to(self.device)
                        concepts, _ = self.model(img)
                        concepts_reshaped = concepts.reshape(1, 7, 3)
                        intervened = concepts_reshaped.clone()
                        
                        for j in wrong_concepts:
                            gt_class = int(concept_labels[i, j])
                            intervened[0, j, :] = 0.0
                            intervened[0, j, gt_class] = 1.0
                        
                        intervened_flat = intervened.reshape(1, -1)
                        intervened_logits = self.model.task_predictor(intervened_flat)
                        cumulative_probs = torch.softmax(intervened_logits, dim=1).cpu().numpy()
                    
                    cumulative_prob = cumulative_probs[0, 1]
                    cumulative_pred = int(cumulative_prob > 0.5)
                    
                    record['cumulative_intervention_prob'] = cumulative_prob
                    record['cumulative_intervention_pred'] = cumulative_pred
                    record['cumulative_intervention_change'] = cumulative_prob - original_melanoma_prob[i]
                    record['cumulative_intervention_fixes'] = (
                        (original_pred[i] != labels[i]) and 
                        (cumulative_pred == labels[i])
                    )
                    record['num_wrong_concepts'] = len(wrong_concepts)
                else:
                    record['cumulative_intervention_prob'] = np.nan
                    record['cumulative_intervention_pred'] = np.nan
                    record['cumulative_intervention_change'] = np.nan
                    record['cumulative_intervention_fixes'] = False
                    record['num_wrong_concepts'] = 0
                
                all_results.append(record)
        
        # Create DataFrames
        df_full = pd.DataFrame(all_results)
        
        # Save full results
        df_full.to_csv(os.path.join(output_dir, 'full_intervention_results.csv'), index=False)
        print(f"✓ Saved full results to {output_dir}/full_intervention_results.csv")
        
        # Create summary tables
        summaries = self._create_summary_tables(df_full, output_dir)
        
        # Create visualizations
        self._create_visualizations(df_full, output_dir)
        
        return {'full': df_full, **summaries}
    
    def _create_summary_tables(self, df: pd.DataFrame, output_dir: str) -> Dict[str, pd.DataFrame]:
        """Create summary statistics tables."""
        summaries = {}
        
        # 1. Overall performance summary
        overall = {
            'Total Samples': len(df),
            'Original Accuracy': df['correct'].mean(),
            'Avg Melanoma Prob': df['original_prob'].mean(),
            'Avg Wrong Concepts': df['num_wrong_concepts'].mean(),
        }
        
        # Add cumulative intervention stats
        has_cumulative = df['cumulative_intervention_fixes'].notna()
        if has_cumulative.any():
            overall['Cumulative Fixes Rate'] = df.loc[has_cumulative, 'cumulative_intervention_fixes'].mean()
            overall['Avg Cumulative Change'] = df.loc[has_cumulative, 'cumulative_intervention_change'].mean()
        
        df_overall = pd.DataFrame([overall])
        df_overall.to_csv(os.path.join(output_dir, 'overall_summary.csv'), index=False)
        summaries['overall'] = df_overall
        print(f"✓ Saved overall summary")
        
        # 2. Per-concept intervention impact
        concept_stats = []
        for name in self.concept_names:
            intervention_col = f'{name}_intervention_change'
            fixes_col = f'{name}_intervention_fixes'
            correct_col = f'{name}_correct'
            
            # Only consider cases where intervention was possible (concept was wrong)
            intervened = df[intervention_col].notna()
            
            if intervened.any():
                stats = {
                    'Concept': name,
                    'Accuracy': df[correct_col].mean(),
                    'Num Interventions': intervened.sum(),
                    'Avg Prob Change': df.loc[intervened, intervention_col].mean(),
                    'Std Prob Change': df.loc[intervened, intervention_col].std(),
                    'Fixes Diagnosis': df.loc[intervened, fixes_col].sum(),
                    'Fix Rate': df.loc[intervened, fixes_col].mean(),
                    'Max Positive Change': df.loc[intervened, intervention_col].max(),
                    'Max Negative Change': df.loc[intervened, intervention_col].min(),
                }
                concept_stats.append(stats)
        
        df_concepts = pd.DataFrame(concept_stats)
        df_concepts = df_concepts.sort_values('Avg Prob Change', ascending=False, key=abs)
        df_concepts.to_csv(os.path.join(output_dir, 'per_concept_impact.csv'), index=False)
        summaries['concepts'] = df_concepts
        print(f"✓ Saved per-concept impact summary")
        
        # 3. Misclassified cases analysis
        misclassified = df[~df['correct']].copy()
        if len(misclassified) > 0:
            misc_stats = {
                'Total Misclassified': len(misclassified),
                'Avg Wrong Concepts': misclassified['num_wrong_concepts'].mean(),
                'Fixed by Cumulative': misclassified['cumulative_intervention_fixes'].sum(),
                'Fix Rate': misclassified['cumulative_intervention_fixes'].mean(),
            }
            df_misc = pd.DataFrame([misc_stats])
            df_misc.to_csv(os.path.join(output_dir, 'misclassified_summary.csv'), index=False)
            summaries['misclassified'] = df_misc
            print(f"✓ Saved misclassified cases summary")
        
        return summaries
    
    def _create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create visualization plots."""
        
        # 1. Performance Improvement: Original vs Intervened (Cumulative)
        has_cumulative = df['cumulative_intervention_change'].notna()
        if has_cumulative.any():
            from sklearn.metrics import f1_score
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # 1a. Comprehensive Performance Metrics (F1, Precision, Recall)
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            y_true_orig = df['true_label']
            y_pred_orig = df['original_pred']
            y_true_int = df.loc[has_cumulative, 'true_label']
            y_pred_int = df.loc[has_cumulative, 'cumulative_intervention_pred']
            
            # Calculate all metrics
            metrics = {
                'F1': [f1_score(y_true_orig, y_pred_orig), f1_score(y_true_int, y_pred_int)],
                'Precision': [precision_score(y_true_orig, y_pred_orig), precision_score(y_true_int, y_pred_int)],
                'Recall': [recall_score(y_true_orig, y_pred_orig), recall_score(y_true_int, y_pred_int)]
            }
            
            ax = axes[0, 0]
            x = np.arange(2)
            width = 0.25
            
            colors_metrics = {'F1': '#2196f3', 'Precision': '#ff9800', 'Recall': '#4caf50'}
            for i, (metric, values) in enumerate(metrics.items()):
                offset = (i - 1) * width
                bars = ax.bar(x + offset, values, width, label=metric, 
                             color=colors_metrics[metric], alpha=0.7, edgecolor='black', linewidth=1.5)
                # Add value labels
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Melanoma Detection Performance\nF1 vs Precision vs Recall', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['Original', 'Intervened'])
            ax.set_ylim([0, 1])
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Baseline')
            
            # 1b. Probability distribution shift
            ax = axes[0, 1]
            original_probs = df.loc[has_cumulative, 'original_prob']
            intervened_probs = df.loc[has_cumulative, 'cumulative_intervention_prob']
            
            ax.hist(original_probs, bins=30, alpha=0.6, label='Original', color='#1976d2', edgecolor='black')
            ax.hist(intervened_probs, bins=30, alpha=0.6, label='Intervened', color='#4caf50', edgecolor='black')
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
            ax.set_xlabel('Melanoma Probability', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Probability Distribution: Before vs After', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            
            # 1c. Fix rate by diagnosis correctness
            ax = axes[1, 0]
            misclassified = df[~df['correct'] & has_cumulative]
            fix_rate = misclassified['cumulative_intervention_fixes'].mean()
            no_fix_rate = 1 - fix_rate
            
            categories = ['Fixed by\nIntervention', 'Still\nMisclassified']
            values = [fix_rate, no_fix_rate]
            colors = ['#4caf50', '#f44336']
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Proportion of Errors', fontsize=12, fontweight='bold')
            ax.set_title(f'Error Recovery Rate\n({len(misclassified)} misclassified cases)', fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 1d. Intervention change by number of wrong concepts
            ax = axes[1, 1]
            if len(misclassified) > 0:
                ax.scatter(misclassified['num_wrong_concepts'], 
                          misclassified['cumulative_intervention_change'],
                          alpha=0.6, s=80, c=misclassified['cumulative_intervention_fixes'].map({True: '#4caf50', False: '#f44336'}),
                          edgecolors='black', linewidth=1)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_xlabel('Number of Wrong Concepts', fontsize=11, fontweight='bold')
                ax.set_ylabel('Probability Change', fontsize=11, fontweight='bold')
                ax.set_title('Intervention Impact vs Concept Errors\n(Green=Fixed, Red=Not Fixed)', fontsize=13, fontweight='bold')
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '1_performance_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved performance comparison plot")
        
        # 2. Per-Concept Intervention Impact (Detailed)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 2a. Box plots of intervention effects
        ax = axes[0, 0]
        impact_data = []
        labels_list = []
        
        for name in self.concept_names:
            intervention_col = f'{name}_intervention_change'
            intervened = df[intervention_col].notna()
            if intervened.any():
                impact_data.append(df.loc[intervened, intervention_col].values)
                labels_list.append(name.replace(' ', '\n'))
        
        bp = ax.boxplot(impact_data, labels=labels_list, patch_artist=True, showfliers=True)
        
        for i, patch in enumerate(bp['boxes']):
            avg_change = np.mean(impact_data[i])
            patch.set_facecolor('#ffcccc' if avg_change > 0 else '#ccccff')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        ax.set_ylabel('Melanoma Probability Change', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Intervention Effects per Concept\n(Red=Increases Melanoma, Blue=Decreases)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=9)
        
        # 2b. Average impact ranking
        ax = axes[0, 1]
        concept_impacts = []
        for name in self.concept_names:
            intervention_col = f'{name}_intervention_change'
            intervened = df[intervention_col].notna()
            if intervened.any():
                avg = df.loc[intervened, intervention_col].mean()
                concept_impacts.append((name, avg))
        
        concept_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        names = [x[0] for x in concept_impacts]
        impacts = [x[1] for x in concept_impacts]
        colors = ['#ff9800' if imp > 0 else '#2196f3' for imp in impacts]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, impacts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax.set_xlabel('Average Probability Change', fontsize=11, fontweight='bold')
        ax.set_title('Concept Impact Ranking\n(Most Influential First)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 2c. Fix rate by concept
        ax = axes[1, 0]
        concept_fix_rates = []
        for name in self.concept_names:
            fixes_col = f'{name}_intervention_fixes'
            intervention_col = f'{name}_intervention_change'
            intervened = df[intervention_col].notna()
            if intervened.any():
                fix_rate = df.loc[intervened, fixes_col].mean()
                concept_fix_rates.append((name, fix_rate, intervened.sum()))
        
        concept_fix_rates.sort(key=lambda x: x[1], reverse=True)
        names = [x[0] for x in concept_fix_rates]
        fix_rates = [x[1] for x in concept_fix_rates]
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, fix_rates, color='#4caf50', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Diagnosis Fix Rate', fontsize=11, fontweight='bold')
        ax.set_title('Concept Intervention Success Rate\n(Proportion of Errors Fixed)', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Add count labels
        for i, (bar, (name, rate, count)) in enumerate(zip(bars, concept_fix_rates)):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                   f'{rate:.1%} (n={count})', va='center', fontsize=9)
        
        # 2d. Concept accuracy vs intervention frequency
        ax = axes[1, 1]
        concept_stats = []
        for name in self.concept_names:
            correct_col = f'{name}_correct'
            intervention_col = f'{name}_intervention_change'
            accuracy = df[correct_col].mean()
            num_interventions = df[intervention_col].notna().sum()
            concept_stats.append((name, accuracy, num_interventions))
        
        names = [x[0] for x in concept_stats]
        accuracies = [x[1] for x in concept_stats]
        intervention_counts = [x[2] for x in concept_stats]
        
        scatter = ax.scatter(accuracies, intervention_counts, s=200, alpha=0.6, 
                           c=range(len(names)), cmap='viridis', edgecolors='black', linewidth=2)
        
        for i, (name, acc, count) in enumerate(concept_stats):
            ax.annotate(name.split()[0], (acc, count), fontsize=8, ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel('Concept Accuracy', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Interventions Needed', fontsize=11, fontweight='bold')
        ax.set_title('Concept Performance: Accuracy vs Error Frequency\n(Lower-right = Most Problematic)', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_concept_impact_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved concept impact analysis plot")
        
        # 3. Intervention Direction Analysis (Positive vs Negative Impact)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 3a. Positive vs Negative changes per concept
        ax = axes[0]
        positive_counts = []
        negative_counts = []
        concept_labels = []
        
        for name in self.concept_names:
            intervention_col = f'{name}_intervention_change'
            intervened = df[intervention_col].notna()
            if intervened.any():
                changes = df.loc[intervened, intervention_col]
                positive_counts.append((changes > 0).sum())
                negative_counts.append((changes < 0).sum())
                concept_labels.append(name.replace(' ', '\n'))
        
        x = np.arange(len(concept_labels))
        width = 0.35
        
        ax.bar(x - width/2, positive_counts, width, label='Increases Melanoma Prob', 
               color='#ff9800', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, negative_counts, width, label='Decreases Melanoma Prob', 
               color='#2196f3', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Concept', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Interventions', fontsize=11, fontweight='bold')
        ax.set_title('Intervention Direction Distribution\n(Effect on Melanoma Probability)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(concept_labels, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # 3b. Magnitude of positive vs negative changes
        ax = axes[1]
        avg_positive = []
        avg_negative = []
        
        for name in self.concept_names:
            intervention_col = f'{name}_intervention_change'
            intervened = df[intervention_col].notna()
            if intervened.any():
                changes = df.loc[intervened, intervention_col]
                pos_changes = changes[changes > 0]
                neg_changes = changes[changes < 0]
                avg_positive.append(pos_changes.mean() if len(pos_changes) > 0 else 0)
                avg_negative.append(neg_changes.mean() if len(neg_changes) > 0 else 0)
        
        x = np.arange(len(concept_labels))
        ax.bar(x - width/2, avg_positive, width, label='Avg Increase', 
               color='#ff9800', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, avg_negative, width, label='Avg Decrease', 
               color='#2196f3', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.set_xlabel('Concept', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Probability Change', fontsize=11, fontweight='bold')
        ax.set_title('Average Magnitude of Intervention Effects\n(Positive vs Negative Impact)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(concept_labels, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_intervention_direction_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved intervention direction analysis plot")
        
        # 4. Confusion matrices: Original vs Intervened
        if has_cumulative.any():
            from sklearn.metrics import confusion_matrix
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original predictions
            ax = axes[0]
            cm_original = confusion_matrix(df['true_label'], df['original_pred'])
            im = ax.imshow(cm_original, cmap='Blues', alpha=0.7)
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Non-Melanoma', 'Melanoma'])
            ax.set_yticklabels(['Non-Melanoma', 'Melanoma'])
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_title(f'Original Predictions\nAccuracy: {df["correct"].mean():.1%}', fontsize=13, fontweight='bold')
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, cm_original[i, j], ha="center", va="center", 
                                 color="black", fontsize=16, fontweight='bold')
            
            # Intervened predictions (cumulative)
            ax = axes[1]
            intervened_labels = df.loc[has_cumulative, 'cumulative_intervention_pred']
            true_labels = df.loc[has_cumulative, 'true_label']
            cm_intervened = confusion_matrix(true_labels, intervened_labels)
            im = ax.imshow(cm_intervened, cmap='Greens', alpha=0.7)
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Non-Melanoma', 'Melanoma'])
            ax.set_yticklabels(['Non-Melanoma', 'Melanoma'])
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            
            intervened_acc = (intervened_labels == true_labels).mean()
            ax.set_title(f'After Cumulative Intervention\nAccuracy: {intervened_acc:.1%}', fontsize=13, fontweight='bold')
            
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, cm_intervened[i, j], ha="center", va="center", 
                                 color="black", fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '4_confusion_matrices.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved confusion matrices plot")
        
        print(f"\n✓ All visualizations saved to {output_dir}/")


def main():
    """Main function for running intervention analysis."""
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description='Concept intervention analysis on test set')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs/intervention_analysis',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    from src.models.basic_cbm import ConceptBottleneckModel
    model = ConceptBottleneckModel.load(args.model_path)
    
    # Load test dataset
    print(f"Loading test dataset from {args.data_dir}...")
    from src.data.derm7pt_adapter import Derm7ptCBMAdapter
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = Derm7ptCBMAdapter(
        data_path=args.data_dir,
        split='test',
        image_size=(224, 224)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run analysis
    analyzer = InterventionAnalyzer(model)
    results = analyzer.analyze_test_set(test_loader, args.output_dir)
    
    print("\n" + "="*70)
    print("Intervention Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - full_intervention_results.csv: Complete results for all samples")
    print("  - overall_summary.csv: Overall performance metrics")
    print("  - per_concept_impact.csv: Impact of each concept intervention")
    print("  - misclassified_summary.csv: Analysis of misclassified cases")
    print("  - intervention_effects_distribution.png: Box plot of intervention effects")
    print("  - cumulative_intervention_analysis.png: Cumulative intervention analysis")


if __name__ == '__main__':
    main()
