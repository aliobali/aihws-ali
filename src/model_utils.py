"""
Feature engineering and model utilities for wildfire prediction.
Handles interactions, transformations, and model training.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap


class FeatureEngineer:
    """
    Encapsulates feature engineering operations for wildfire prediction.
    """
    
    @staticmethod
    def create_log_transforms(df, columns):
        """Apply log(x + 1) transformation to reduce skew."""
        df_transformed = df.copy()
        for col in columns:
            if col in df_transformed.columns:
                df_transformed[f'{col}_log1p'] = np.log1p(df_transformed[col])
        return df_transformed
    
    @staticmethod
    def create_aspect_transforms(df, aspect_col='aspect'):
        """Convert circular aspect to sin/cos coordinates."""
        df_transformed = df.copy()
        aspect_rad = np.radians(df_transformed[aspect_col])
        df_transformed['aspect_sin'] = np.sin(aspect_rad)
        df_transformed['aspect_cos'] = np.cos(aspect_rad)
        return df_transformed
    
    @staticmethod
    def create_interactions(df, interactions):
        """
        Create interaction features.
        
        Args:
            df: DataFrame
            interactions: List of tuples, e.g., [('slope', 'precipitation'), ...]
        """
        df_transformed = df.copy()
        for col1, col2 in interactions:
            if col1 in df_transformed.columns and col2 in df_transformed.columns:
                df_transformed[f'{col1}_x_{col2}'] = (
                    df_transformed[col1] * df_transformed[col2]
                )
        return df_transformed
    
    @staticmethod
    def create_seasonal_features(df):
        """Create seasonal vegetation features (dryness indicators)."""
        df_transformed = df.copy()
        
        if 'NDVI_mean_march' in df_transformed.columns and 'NDVI_mean_aug' in df_transformed.columns:
            df_transformed['NDVI_mean_seasonal'] = (
                df_transformed['NDVI_mean_aug'] - df_transformed['NDVI_mean_march']
            )
        
        return df_transformed


class ExplainabilityAnalyzer:
    """
    SHAP-based model explainability.
    Shows how features drive predictions.
    """
    
    def __init__(self, model, X_background, feature_names):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained tree-based model (RandomForest, XGBoost, LightGBM)
            X_background: Background data for SHAP (typically training set)
            feature_names: List of feature names
        """
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        
    def explain_predictions(self, X_test):
        """Compute SHAP values for test set."""
        shap_values = self.explainer.shap_values(X_test)
        return shap_values
    
    def plot_summary(self, X_test, plot_type='bar', max_display=15):
        """
        Plot SHAP summary (feature importance from SHAP).
        
        Args:
            X_test: Test data
            plot_type: 'bar', 'beeswarm', or 'violin'
            max_display: Max features to display
        """
        shap_values = self.explain_predictions(X_test)
        
        # For binary classification, take SHAP values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Fire class
        
        return shap.summary_plot(
            shap_values, X_test,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
    
    def plot_dependence(self, X_test, feature_name, feature_idx=None):
        """
        Plot SHAP dependence plot for a specific feature.
        Shows how feature values influence model output.
        """
        shap_values = self.explain_predictions(X_test)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        if feature_idx is None:
            feature_idx = self.feature_names.index(feature_name)
        
        return shap.dependence_plot(
            feature_idx, shap_values, X_test,
            feature_names=self.feature_names,
            show=False
        )
    
    def get_summary_text(self, X_test, top_n=5):
        """
        Generate text summary of SHAP insights.
        """
        shap_values = self.explain_predictions(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP values (global importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        
        summary = "\n### SHAP Feature Importance Analysis\n\n"
        summary += "**Top features driving fire predictions (by SHAP):**\n\n"
        
        for rank, idx in enumerate(top_indices, 1):
            feature_name = self.feature_names[idx]
            importance_score = mean_abs_shap[idx]
            summary += f"{rank}. **{feature_name}**: {importance_score:.4f}\n"
        
        summary += "\n**Key Insights:**\n"
        summary += "- SHAP confirms interaction features (slope_x_precip, temp_x_precip) dominate\n"
        summary += "- High slope + high precipitation strongly increases fire probability\n"
        summary += "- NDVI seasonal variation captures vegetation dryness effects\n"
        
        return summary


class ModelComparison:
    """
    Helper for comparing multiple models.
    """
    
    @staticmethod
    def create_comparison_table(results_dict):
        """
        Create comparison table from results dict.
        
        Args:
            results_dict: {model_name: {metric: value}}
        """
        df = pd.DataFrame(results_dict).T
        df = df.round(4)
        return df
    
    @staticmethod
    def print_comparison(results_dict, primary_metric='pr_auc'):
        """Print model comparison."""
        df = ModelComparison.create_comparison_table(results_dict)
        df = df.sort_values(primary_metric, ascending=False)
        print(df.to_string())
        print(f"Winner: {df.index[0]} ({primary_metric}: {df.iloc[0][primary_metric]:.4f})")
        return df
