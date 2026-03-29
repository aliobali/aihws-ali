"""
Evaluation utilities for wildfire classification and regression models.
Focus on PR-AUC due to class imbalance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score, 
    roc_auc_score,
    confusion_matrix, 
    classification_report, 
    f1_score, 
    accuracy_score,
    precision_score,
    recall_score
)


class ClassificationEvaluator:
    """
    Evaluator for classification tasks with focus on imbalanced data.
    Uses PR-AUC as primary metric (not ROC-AUC).
    """
    
    def __init__(self, y_true, y_pred, y_pred_proba, class_names=["No Fire", "Fire"]):
        """
        Initialize evaluator.
        
        Args:
            y_true: Ground truth labels
            y_pred: Binary predictions
            y_pred_proba: Probability predictions [0, 1]
            class_names: Names of classes
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.class_names = class_names
    
    def compute_metrics(self):
        """
        Compute all evaluation metrics.
        Returns dict with key metrics.
        """
        metrics = {
            # Primary metric for imbalanced data
            'pr_auc': average_precision_score(self.y_true, self.y_pred_proba),
            
            # Secondary metrics
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1': f1_score(self.y_true, self.y_pred),
            
            # Deprecated for class imbalance but included for reference
            'roc_auc': roc_auc_score(self.y_true, self.y_pred_proba),
        }
        return metrics
    
    def print_summary(self):
        """Print evaluation summary."""
        metrics = self.compute_metrics()
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        return metrics
    
    def plot_confusion_matrix(self, figsize=(8, 6)):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'},
                   ax=ax)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix\n(Fire Occurrence Classification)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add percentages
        total = cm.sum()
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                pct = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                       ha='center', va='center', fontsize=9, color='darkgray')
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, figsize=(10, 7)):
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_true, self.y_pred_proba)
        baseline = np.mean(self.y_true)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, linewidth=3, label=f'Model (PR-AUC = {pr_auc:.3f})', color='#2E86AB')
        ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline = {baseline:.3f}')
        ax.fill_between(recall, precision, alpha=0.2, color='#2E86AB')
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        return fig
    
    def plot_threshold_tradeoff(self, figsize=(12, 6)):
        """Show Precision, Recall, F1 vs decision threshold."""
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(thresholds, precision[:-1], 'o-', linewidth=2, label='Precision', color='#06A77D')
        ax.plot(thresholds, recall[:-1], 's-', linewidth=2, label='Recall', color='#D62246')
        ax.plot(thresholds, f1_scores, '^-', linewidth=2, label='F1-Score', color='#F77F00')
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, label='Default (0.5)')
        ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall-F1 vs Threshold', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        return fig


def plot_feature_importance_clean(feature_importances, feature_names, 
                                  top_n=15, figsize=(12, 8)):
    """
    Clean, publication-ready feature importance plot.
    
    Args:
        feature_importances: Array of importance scores
        feature_names: List of feature names
        top_n: Number of top features to show
        figsize: Figure size
    """
    # Get top N indices
    indices = np.argsort(feature_importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(indices)))
    bars = ax.barh(range(len(indices)), 
                   feature_importances[indices],
                   color=colors,
                   edgecolor='black',
                   linewidth=1.2)
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, bar) in enumerate(zip(indices, bars)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {width:.4f}', 
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_correlation_matrix_clean(df, figsize=(14, 12), top_n_features=None):
    """
    Clean correlation matrix heatmap with readable labels.
    
    Args:
        df: DataFrame
        figsize: Figure size
        top_n_features: If specified, select top N features by variance
    """
    # Select only numeric columns (exclude categorical)
    df_numeric = df.select_dtypes(include=['number'])
    
    if top_n_features:
        # Select top features by variance
        top_features = df_numeric.var().nlargest(top_n_features).index
        df_numeric = df_numeric[top_features]
    
    corr = df_numeric.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr, cmap='RdBu_r', center=0, 
               square=True, linewidths=0.5,
               cbar_kws={'label': 'Correlation', 'shrink': 0.8},
               xticklabels=True, yticklabels=True,
               ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_title('Feature Correlation Matrix', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig
