#!/usr/bin/env python
"""SHAP feature explainability analysis."""

import json
import joblib
import shap
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def main():
    print("SHAP Feature Explainability Analysis\n")
    
    # Load data and model
    project_root = Path(__file__).parent.parent
    df = pd.read_csv(project_root / 'data' / 'wildfire_data_for_training.csv')
    
    with open(project_root / 'models' / 'features.json') as f:
        feature_names = json.load(f)
    
    clf = joblib.load(project_root / 'models' / 'clf_occurrence.pkl')
    print(f"Loaded {len(feature_names)} features")
    
    # Prepare data
    y = df['fire_occurrence'].values
    X = df[[col for col in feature_names if col in df.columns]].copy()
    X = SimpleImputer(strategy='median').fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    
    # Limit to 1000 for memory efficiency
    X_test = X_test[:1000]
    y_test = y_test[:1000]
    print(f"Test set: {X_test.shape[0]:,} samples\n")
    
    # SHAP TreeExplainer
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values_full = explainer.shap_values(X_test)
    
    # For binary classification with shape (n_samples, n_features, 2), 
    # extract the positive class (index 1)
    if isinstance(shap_values_full, list):
        shap_values = shap_values_full[1]
    elif len(shap_values_full.shape) == 3:
        # Shape (n_samples, n_features, 2) -> take positive class
        shap_values = shap_values_full[:, :, 1]
        
    # Handle expected_value
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value = explainer.expected_value[1]
    else:
        expected_value = explainer.expected_value
    
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Expected value: {expected_value}")
    
    # 1. Bar plot - summary importance
    print("Generating SHAP bar plot...")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.savefig(project_root / 'outputs' / 'shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Beeswarm plot - summary with instance details
    print("Generating SHAP beeswarm plot...")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='violin', show=False)
    plt.savefig(project_root / 'outputs' / 'shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Waterfall plots - explain 2-3 individual predictions
    print("Generating SHAP waterfall plots...")
    # Find fire and non-fire examples
    fire_idx = np.where(y_test == 1)[0]
    no_fire_idx = np.where(y_test == 0)[0]
    
    examples = []
    if len(fire_idx) > 0:
        examples.append((fire_idx[0], "Fire"))  # First fire prediction
    if len(no_fire_idx) > 0:
        examples.append((no_fire_idx[0], "No Fire"))  # First non-fire prediction
    if len(fire_idx) > 1:
        examples.append((fire_idx[1], "Fire"))  # Second fire prediction
    
    # Create individual waterfall plots
    for plot_idx, (idx, label) in enumerate(examples):
        plt.figure(figsize=(12, 4))
        shap.waterfall_plot(shap.Explanation(values=shap_values[idx], 
                                             base_values=expected_value, 
                                             data=X_test[idx], 
                                             feature_names=feature_names),
                           max_display=10, show=False)
        plt.suptitle(f"Prediction {plot_idx+1}: {label}", fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(project_root / 'outputs' / f'shap_waterfall_example_{plot_idx+1}_{label.replace(" ", "_").lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Generated {len(examples)} waterfall plots")
    
    # 4. Dependence plots - top 3 features
    print("Generating SHAP dependence plots...")
    importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(importance)[-3:][::-1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, idx in enumerate(top_idx):
        ax = axes[i]
        try:
            shap.dependence_plot(int(idx), shap_values, X_test, 
                               feature_names=feature_names, ax=ax, show=False)
        except Exception as e:
            print(f"Warning: Could not generate dependence plot for feature {idx}: {e}")
    
    plt.tight_layout()
    plt.savefig(project_root / 'outputs' / 'shap_dependence_top3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nDone! Generated outputs:")
    print("  - shap_summary_bar.png (global feature importance)")
    print("  - shap_summary_beeswarm.png (feature impact distribution)")
    print("  - shap_waterfall_example_*.png (individual predictions explained)")
    print("  - shap_dependence_top3.png (feature value interactions)")


if __name__ == '__main__':
    main()



