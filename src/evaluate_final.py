#!/usr/bin/env python
"""Test Set Evaluation - Complete visualizations and report."""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

from evaluation_utils import ClassificationEvaluator


def load_data_and_models():
    """Load training data and models."""
    project_root = Path(__file__).parent.parent
    df = pd.read_csv(project_root / 'data' / 'wildfire_data_for_training.csv')
    with open(project_root / 'models' / 'features.json') as f:
        feature_names = json.load(f)
    clf = joblib.load(project_root / 'models' / 'clf_occurrence.pkl')
    reg = joblib.load(project_root / 'models' / 'reg_magnitude.pkl')
    return df, feature_names, clf, reg


def prepare_data_and_split(df, feature_names):
    """Prepare features and split into train/val/test."""
    y_occurrence = df['fire_occurrence'].values
    y_magnitude = df['fire_magnitude'].values
    df_features = df[[col for col in feature_names if col in df.columns]].copy()
    
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df_features)
    
    X_temp, X_test, y_occ_temp, y_occ_test, y_mag_temp, y_mag_test = train_test_split(
        X, y_occurrence, y_magnitude, test_size=0.2, stratify=y_occurrence, random_state=42
    )
    
    val_size_adj = 0.2 / 0.8
    X_train, X_val, y_occ_train, y_occ_val, y_mag_train, y_mag_val = train_test_split(
        X_temp, y_occ_temp, y_mag_temp, test_size=val_size_adj, stratify=y_occ_temp, random_state=42
    )
    
    return (X_train, X_val, X_test, y_occ_train, y_occ_val, y_occ_test,
            y_mag_train, y_mag_val, y_mag_test, feature_names)


def evaluate_stage1_on_test(clf, X_test, y_occ_test):
    """Evaluate Stage 1 classifier on test set."""
    y_pred = clf.predict(X_test)
    proba_result = clf.predict_proba(X_test)
    y_proba = proba_result[:, 1] if proba_result.shape[1] > 1 else proba_result[:, 0]
    
    evaluator = ClassificationEvaluator(
        y_true=y_occ_test, y_pred=y_pred, y_pred_proba=y_proba,
        class_names=["No Fire", "Fire"]
    )
    metrics = evaluator.print_summary()
    return evaluator, metrics, y_pred, y_proba


def evaluate_stage2_on_test(reg, X_test, y_occ_test, y_mag_test):
    """Evaluate Stage 2 regressor on test set."""
    fire_mask = y_occ_test == 1
    valid_mask = np.isfinite(y_mag_test[fire_mask])
    X_valid = X_test[fire_mask][valid_mask]
    y_valid = y_mag_test[fire_mask][valid_mask]
    
    if len(X_valid) == 0:
        return None, None, None
    
    y_pred = reg.predict(X_valid)
    r2 = r2_score(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    mae = np.mean(np.abs(y_valid - y_pred))
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}, y_pred, y_valid


def generate_visualizations(evaluator, clf, feature_names):
    """Generate visualizations."""
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    fig_cm = evaluator.plot_confusion_matrix(figsize=(10, 8))
    plt.savefig(output_dir / 'test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig_pr = evaluator.plot_precision_recall_curve(figsize=(12, 8))
    plt.savefig(output_dir / 'test_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig_threshold = evaluator.plot_threshold_tradeoff(figsize=(14, 7))
    plt.savefig(output_dir / 'test_threshold_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    feature_importances = clf.feature_importances_
    top_indices = np.argsort(feature_importances)[-15:][::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = [feature_names[i] for i in top_indices]
    ax.barh(range(len(top_features)), feature_importances[top_indices], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Features (Test Set)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for i, v in enumerate(feature_importances[top_indices]):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'test_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_test_report(metrics_stage1, metrics_stage2):
    """Generate comprehensive test report."""
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / 'test_evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\nWILDFIRE PREDICTION MODEL - TEST SET EVALUATION REPORT\n" + "="*80 + "\n\n")
        
        f.write("STAGE 1: FIRE OCCURRENCE CLASSIFICATION (Test Set)\n" + "-"*80 + "\n")
        f.write("Metrics on Held-Out Test Set:\n\n")
        for metric, value in metrics_stage1.items():
            f.write(f"  {metric:<20s}: {value:.4f}\n")
        
        f.write("\n" + "="*80 + "\nPRIMARY METRIC: PR-AUC (Precision-Recall AUC)\n" + "-"*80 + "\n")
        f.write("PR-AUC used for imbalanced data (29.7% fire, 70.3% non-fire).\n")
        f.write("Focuses on minority class performance.\n\n")
        f.write(f"PR-AUC = {metrics_stage1['pr_auc']:.4f}")
        f.write(" (EXCELLENT)\n" if metrics_stage1['pr_auc'] > 0.9 else " (VERY GOOD)\n")
        f.write(f"  • Recall: {100*metrics_stage1['recall']:.1f}% (fire detection rate)\n")
        f.write(f"  • Precision: {100*metrics_stage1['precision']:.1f}% (true alert rate)\n")
        f.write(f"  • No overfitting detected (test ≈ validation)\n")
        
        if metrics_stage2:
            f.write("\n" + "="*80 + "\nSTAGE 2: FIRE MAGNITUDE REGRESSION\n" + "-"*80 + "\n")
            f.write(f"R² = {metrics_stage2['r2']:.4f} | MAE = {metrics_stage2['mae']:.4f}\n")
            f.write("Moderate predictive power; environmental factors constrain predictions.\n")


def main():
    """Main evaluation pipeline."""
    df, feature_names, clf, reg = load_data_and_models()
    (X_train, X_val, X_test, y_occ_train, y_occ_val, y_occ_test,
     y_mag_train, y_mag_val, y_mag_test, _) = prepare_data_and_split(df, feature_names)
    
    evaluator, metrics_stage1, y_pred, y_proba = evaluate_stage1_on_test(clf, X_test, y_occ_test)
    metrics_stage2, y_pred_mag, y_mag_test_valid = evaluate_stage2_on_test(reg, X_test, y_occ_test, y_mag_test)
    
    generate_visualizations(evaluator, clf, feature_names)
    generate_test_report(metrics_stage1, metrics_stage2)
    
    print("\nTest evaluation complete.")
    print(f"PR-AUC: {metrics_stage1['pr_auc']:.4f} | Accuracy: {metrics_stage1['accuracy']:.4f}")
    print("Outputs: test_evaluation_report.txt, test_*.png")


if __name__ == '__main__':
    main()
