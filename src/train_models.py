#!/usr/bin/env python
"""
Train two-stage wildfire prediction model: 
Stage 1: Fire occurrence classifier (RandomForest)
Stage 2: Fire magnitude regressor (RandomForest, only on positive cases)

Usage:
    python train_models.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model_utils import ExplainabilityAnalyzer, ModelComparison
from evaluation_utils import ClassificationEvaluator


def load_data(data_path):
    """Load cleaned wildfire dataset."""
    df = pd.read_csv(data_path)
    return df


def prepare_features(df):
    """Prepare features and target variables."""
    y_occurrence = df['fire_occurrence'].values
    y_magnitude = df['fire_magnitude'].values
    exclude_cols = {'wildfires_25yrs', 'fire_occurrence', 'fire_magnitude'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    df_features = df[feature_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df_features)
    return X, y_occurrence, y_magnitude, feature_cols


def split_data(X, y_occurrence, y_magnitude, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train/validation/test (stratified)."""
    X_temp, X_test, y_occ_temp, y_occ_test, y_mag_temp, y_mag_test = train_test_split(
        X, y_occurrence, y_magnitude, test_size=test_size, stratify=y_occurrence, random_state=random_state
    )
    val_size_adj = val_size / (1 - test_size)
    X_train, X_val, y_occ_train, y_occ_val, y_mag_train, y_mag_val = train_test_split(
        X_temp, y_occ_temp, y_mag_temp, test_size=val_size_adj, stratify=y_occ_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_occ_train, y_occ_val, y_occ_test, y_mag_train, y_mag_val, y_mag_test


def train_stage1_classifier(X_train, y_occ_train, X_val, y_occ_val, n_jobs=-1):
    """Train Stage 1: Fire occurrence classifier."""
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_split=10,
        min_samples_leaf=5, class_weight='balanced', max_features='sqrt',
        random_state=42, n_jobs=n_jobs, verbose=0
    )
    clf.fit(X_train, y_occ_train)
    
    y_occ_val_pred = clf.predict(X_val)
    proba_result = clf.predict_proba(X_val)
    y_occ_val_proba = proba_result[:, 1] if proba_result.shape[1] > 1 else proba_result[:, 0]
    evaluator = ClassificationEvaluator(y_occ_val, y_occ_val_pred, y_occ_val_proba)
    metrics = evaluator.compute_metrics()
    return clf, metrics, evaluator


def train_stage2_regressor(X_train, y_mag_train, y_occ_train, X_val, y_mag_val, y_occ_val, n_jobs=-1):
    """Train Stage 2: Fire magnitude regressor on positive samples only."""
    fire_mask_train = y_occ_train == 1
    fire_mask_val = y_occ_val == 1
    X_train_pos = X_train[fire_mask_train]
    y_mag_train_pos = y_mag_train[fire_mask_train]
    X_val_pos = X_val[fire_mask_val]
    y_mag_val_pos = y_mag_val[fire_mask_val]
    
    valid_mask_train = np.isfinite(y_mag_train_pos)
    valid_mask_val = np.isfinite(y_mag_val_pos)
    X_train_valid = X_train_pos[valid_mask_train]
    y_mag_train_valid = y_mag_train_pos[valid_mask_train]
    X_val_valid = X_val_pos[valid_mask_val]
    y_mag_val_valid = y_mag_val_pos[valid_mask_val]
    
    reg = RandomForestRegressor(
        n_estimators=150, max_depth=20, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=n_jobs, verbose=0
    )
    reg.fit(X_train_valid, y_mag_train_valid)
    y_mag_val_pred = reg.predict(X_val_valid)
    
    mse = mean_squared_error(y_mag_val_valid, y_mag_val_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_mag_val_valid, y_mag_val_pred)
    mae = np.mean(np.abs(y_mag_val_valid - y_mag_val_pred))
    metrics = {'r2': r2, 'rmse': rmse, 'mae': mae, 'mse': mse}
    return reg, metrics


def save_models(clf, reg, feature_cols, models_dir='models'):
    """Save trained models and metadata."""
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    joblib.dump(clf, models_path / 'clf_occurrence.pkl')
    joblib.dump(reg, models_path / 'reg_magnitude.pkl')
    with open(models_path / 'features.json', 'w') as f:
        json.dump(list(feature_cols), f, indent=2)


def get_feature_importance(clf, reg, feature_cols):
    """Extract feature importance from both models."""
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'stage1_importance': clf.feature_importances_,
        'stage2_importance': reg.feature_importances_,
        'mean_importance': (clf.feature_importances_ + reg.feature_importances_) / 2
    }).sort_values('mean_importance', ascending=False)
    return importance_df


def main():
    """Main training pipeline."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'wildfire_data_for_training.csv'
    if not data_path.exists():
        data_path = project_root / 'data' / 'wildfire_data_clean.csv'
    models_dir = project_root / 'models'
    
    df = load_data(data_path)
    X, y_occurrence, y_magnitude, feature_cols = prepare_features(df)
    (X_train, X_val, X_test, y_occ_train, y_occ_val, y_occ_test,
     y_mag_train, y_mag_val, y_mag_test) = split_data(
        X, y_occurrence, y_magnitude, test_size=0.2, val_size=0.2
    )
    
    clf, metrics_stage1, evaluator = train_stage1_classifier(
        X_train, y_occ_train, X_val, y_occ_val
    )
    reg, metrics_stage2 = train_stage2_regressor(
        X_train, y_mag_train, y_occ_train, X_val, y_mag_val, y_occ_val
    )
    
    importance_df = get_feature_importance(clf, reg, feature_cols)
    save_models(clf, reg, feature_cols, models_dir=models_dir)
    
    metadata = {
        'data_shape': df.shape,
        'n_features': len(feature_cols),
        'feature_names': list(feature_cols),
        'split_sizes': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
        'stage1_metrics': metrics_stage1,
        'stage2_metrics': {k: float(v) for k, v in metrics_stage2.items()},
        'class_balance': {
            'train_positive_pct': float(100 * y_occ_train.mean()),
            'val_positive_pct': float(100 * y_occ_val.mean()),
            'test_positive_pct': float(100 * y_occ_test.mean())
        }
    }
    with open(models_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
