#!/usr/bin/env python
"""SHAP feature explainability analysis."""

import json
import joblib
import shap
import pandas as pd
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
    
    # Limit to 2000 for speed
    X_test = X_test[:2000]
    print(f"Test set: {X_test.shape[0]:,} samples\n")
    
    # SHAP TreeExplainer
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Save summary plot
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.savefig(project_root / 'outputs' / 'shap_summary_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Done! Outputs saved to outputs/")


if __name__ == '__main__':
    main()



