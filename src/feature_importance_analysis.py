#!/usr/bin/env python
"""Feature Importance Analysis using RandomForest + Permutation Importance."""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')


def load_data_and_model():
    """Load data and trained classifier."""
    project_root = Path(__file__).parent.parent
    df = pd.read_csv(project_root / 'data' / 'wildfire_data_for_training.csv')
    with open(project_root / 'models' / 'features.json') as f:
        feature_names = json.load(f)
    clf = joblib.load(project_root / 'models' / 'clf_occurrence.pkl')
    return df, feature_names, clf


def prepare_data(df, feature_names):
    """Prepare and split data."""
    y_occurrence = df['fire_occurrence'].values
    df_features = df[[col for col in feature_names if col in df.columns]].copy()
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df_features)
    
    X_temp, X_test, y_occ_temp, y_occ_test = train_test_split(
        X, y_occurrence, test_size=0.2, stratify=y_occurrence, random_state=42
    )
    val_size_adj = 0.2 / 0.8
    X_train, X_val, y_occ_train, y_occ_val = train_test_split(
        X_temp, y_occ_temp, test_size=val_size_adj, stratify=y_occ_temp, random_state=42
    )
    return X_train, X_val, X_test, y_occ_train, y_occ_val, y_occ_test, feature_names


def generate_feature_importance_analysis(clf, X_train, X_test, y_occ_test, feature_names):
    """Generate feature importance using multiple methods."""
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    rf_importance = clf.feature_importances_
    perm_result = permutation_importance(
        clf, X_test, y_occ_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_importance = perm_result.importances_mean
    
    rf_norm = rf_importance / rf_importance.max()
    perm_norm = perm_importance / perm_importance.max()
    combined_importance = (rf_norm + perm_norm) / 2
    top_indices = np.argsort(combined_importance)[-15:][::-1]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'RF Importance': rf_importance,
        'Permutation Importance': perm_importance,
        'Combined Score': combined_importance
    })
    
    return importance_df, rf_importance, perm_importance, combined_importance, feature_names, top_indices


def generate_visualizations(importance_df, rf_importance, perm_importance, combined_importance, feature_names, top_indices):
    """Generate comparison visualizations."""
    output_dir = Path('outputs')
    top_features = [feature_names[i] for i in top_indices]
    top_combined = combined_importance[top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_features)), top_combined, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - Combined Analysis\n(RandomForest + Permutation)', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    for i, v in enumerate(top_combined):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top_features))
    width = 0.35
    top_rf = rf_importance[top_indices]
    top_perm = perm_importance[top_indices]
    ax.barh(x - width/2, top_rf / top_rf.max(), width, label='RandomForest Importance', color='steelblue', alpha=0.8)
    ax.barh(x + width/2, top_perm / top_perm.max(), width, label='Permutation Importance', color='coral', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(top_features, fontsize=11)
    ax.set_xlabel('Normalized Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Methods Comparison\n(Top 15 Features)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    importance_df_sorted = importance_df.sort_values('Combined Score', ascending=False)
    importance_df_sorted.to_csv(output_dir / 'feature_importance_detailed.csv', index=False)


def generate_report(importance_df, top_indices, feature_names, rf_importance, perm_importance, combined_importance):
    """Generate comprehensive report."""
    output_dir = Path('outputs')
    report_path = output_dir / 'feature_importance_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\nFEATURE IMPORTANCE ANALYSIS REPORT\nFire Occurrence Prediction Model\n" + "="*80 + "\n\n")
        
        f.write("METHODOLOGY\n" + "-"*80 + "\n")
        f.write("1. RandomForest Feature Importance - Gini-based built-in method\n")
        f.write("2. Permutation Importance - Model-agnostic, measures performance impact\n")
        f.write("3. Combined Score - Average of both methods (normalized to 0-1)\n\n")
        
        f.write("="*80 + "\nTOP 15 MOST PREDICTIVE FEATURES\n" + "="*80 + "\n\n")
        
        for rank, idx in enumerate(top_indices, 1):
            feat = feature_names[idx]
            rf_imp = rf_importance[idx]
            perm_imp = perm_importance[idx]
            combined = combined_importance[idx]
            f.write(f"{rank:2d}. {feat:<30s}\n")
            f.write(f"    Combined Score: {combined:.5f}\n")
            f.write(f"    RF Importance:  {rf_imp:.5f}\n")
            f.write(f"    Perm Importance:{perm_imp:.5f}\n\n")
        
        f.write("="*80 + "\nKEY INSIGHTS\n" + "="*80 + "\n\n")
        f.write(f"1. Most Predictive: {feature_names[top_indices[0]]} (score: {combined_importance[top_indices[0]]:.5f})\n\n")
        
        top_5 = [feature_names[i] for i in top_indices[:5]]
        f.write("2. Top 5 Features:\n")
        for i, feat in enumerate(top_5, 1):
            idx = feature_names.index(feat)
            f.write(f"   {i}. {feat} ({combined_importance[idx]:.5f})\n")
        f.write("\n")
        
        interaction_terms = [feature_names[i] for i in top_indices if '_x_' in feature_names[i]]
        f.write(f"3. Interaction Terms in Top 15: {len(interaction_terms)}\n")
        if interaction_terms:
            f.write(f"   Examples: {', '.join(interaction_terms[:3])}\n")
        f.write("   Model learns complex relationships beyond individual factors.\n")


def main():
    """Main analysis pipeline."""
    df, feature_names, clf = load_data_and_model()
    X_train, X_val, X_test, y_occ_train, y_occ_val, y_occ_test, _ = prepare_data(df, feature_names)
    
    importance_df, rf_importance, perm_importance, combined_importance, feature_names, top_indices = generate_feature_importance_analysis(
        clf, X_train, X_test, y_occ_test, feature_names
    )
    
    generate_visualizations(
        importance_df, rf_importance, perm_importance, combined_importance, feature_names, top_indices
    )
    
    generate_report(
        importance_df, top_indices, feature_names, rf_importance, perm_importance, combined_importance
    )
    
    print("Feature importance analysis complete.")
    print("Outputs: feature_importance_combined.png, feature_importance_comparison.png, feature_importance_detailed.csv")


if __name__ == '__main__':
    main()
