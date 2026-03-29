#!/usr/bin/env python
"""
Wildfire Prediction on New GeoTIFF Rasters

Loads trained models and generates fire occurrence/magnitude predictions for new raster datasets.

Usage:
    python predict_new_geotiffs.py --input_dir data/new_rasters --output_dir outputs/predictions
"""

import argparse
import json
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from pathlib import Path
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings('ignore')


class WildfirePredictor:
    """Load models and generate predictions on new geospatial data."""
    
    # Expected feature order (11 base features used for both imputation and engineering)
    EXPECTED_FEATURES = [
        '1_road_dist', 'LULC_2019', 'NDVI_mean_aug', 'NDVI_mean_march',
        'aspect', 'max_temp_aug', 'mean_precipitation', 'mean_temp',
        'slope', 'soil_silt', 'wildfires_25yrs'
    ]
    
    def __init__(self, models_dir='models'):
        """Initialize with trained models."""
        self.models_dir = Path(models_dir)
        self.clf = joblib.load(self.models_dir / 'clf_occurrence.pkl')
        self.reg = joblib.load(self.models_dir / 'reg_magnitude.pkl')
        with open(self.models_dir / 'features.json') as f:
            self.feature_names = json.load(f)
        self.imputer = SimpleImputer(strategy='median')
    
    def load_and_align_rasters(self, input_dir, reference_shape=(244, 344)):
        """Load rasters in correct feature order and align to common grid."""
        input_dir = Path(input_dir)
        rasters = []
        
        for feature_name in self.EXPECTED_FEATURES:
            # Find the matching raster file
            raster_files = list(input_dir.glob(f'{feature_name}.tif*'))
            if not raster_files:
                raise FileNotFoundError(f"Missing raster: {feature_name}.tif in {input_dir}")
            
            raster_path = raster_files[0]
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                if data.shape != reference_shape:
                    transform = src.transform
                    target_transform = rasterio.transform.from_bounds(
                        src.bounds.left, src.bounds.bottom,
                        src.bounds.right, src.bounds.top,
                        reference_shape[1], reference_shape[0]
                    )
                    resampled = np.zeros(reference_shape, dtype=data.dtype)
                    rasterio.warp.reproject(
                        data, resampled,
                        src_transform=transform,
                        dst_transform=target_transform,
                        resampling=rasterio.warp.Resampling.bilinear
                    )
                    data = resampled
                rasters.append(data.flatten())
        
        return np.column_stack(rasters), reference_shape
    
    def prepare_features(self, X_raw):
        """Prepare raw raster data for prediction, including feature engineering."""
        # Impute missing values first
        X_imputed = self.imputer.fit_transform(X_raw)
        
        # X_imputed should have 11 base features in this order:
        # 0: 1_road_dist, 1: LULC_2019, 2: NDVI_mean_aug, 3: NDVI_mean_march,
        # 4: aspect, 5: max_temp_aug, 6: mean_precipitation, 7: mean_temp,
        # 8: slope, 9: soil_silt, 10: wildfires_25yrs
        
        if X_imputed.shape[1] < 11:
            raise ValueError(f"Expected at least 11 features, got {X_imputed.shape[1]}")
        
        # Extract base features for engineering
        slope = X_imputed[:, 8]
        soil_silt = X_imputed[:, 9]
        aspect = X_imputed[:, 4]
        mean_precip = X_imputed[:, 6]
        mean_temp = X_imputed[:, 7]
        ndvi_aug = X_imputed[:, 2]
        ndvi_march = X_imputed[:, 3]
        
        # Create engineered features with safe handling
        slope_log1p = np.log1p(np.clip(np.abs(slope), 0, 1e6))  # Avoid inf values
        soil_silt_log1p = np.log1p(np.clip(np.abs(soil_silt), 0, 1e6))
        
        # Convert aspect to radians and compute trig functions
        aspect_rad = np.radians(np.clip(aspect, 0, 360))
        aspect_sin = np.sin(aspect_rad)
        aspect_cos = np.cos(aspect_rad)
        
        # Clip interaction terms to reasonable ranges
        slope_x_precip = np.clip(slope * mean_precip, -1e4, 1e4)
        temp_x_precip = np.clip(mean_temp * mean_precip, -1e4, 1e4)
        ndvi_diff_seasonal = np.clip(ndvi_aug - ndvi_march, -5, 5)
        
        # Stack engineered features (keeping original 10 base + adding 7 engineered = 17 total)
        X_engineered = np.column_stack([
            X_imputed[:, :10],  # First 10 base features (excluding wildfires_25yrs)
            slope_log1p,
            soil_silt_log1p,
            aspect_sin,
            aspect_cos,
            slope_x_precip,
            temp_x_precip,
            ndvi_diff_seasonal
        ])
        
        # Final safety check: replace any remaining inf/nan with column median
        for col in range(X_engineered.shape[1]):
            col_data = X_engineered[:, col]
            bad_mask = ~np.isfinite(col_data)
            if bad_mask.any():
                col_median = np.nanmedian(col_data[np.isfinite(col_data)])
                X_engineered[bad_mask, col] = col_median
        
        return X_engineered
    
    def predict_fire_occurrence(self, X):
        """Predict fire occurrence (binary classification)."""
        y_pred = self.clf.predict(X)
        y_proba = self.clf.predict_proba(X)
        fire_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        return y_pred, fire_proba
    
    def predict_fire_magnitude(self, X, fire_mask):
        """Predict fire magnitude (regression on fire-only pixels)."""
        X_fire = X[fire_mask]
        if len(X_fire) == 0:
            return None
        return self.reg.predict(X_fire)
    
    def create_output_rasters(self, predictions, shape, output_dir, basename='predictions'):
        """Save predictions as GeoTIFF rasters."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        y_pred, fire_proba = predictions
        
        occ_map = y_pred.reshape(shape)
        occ_path = output_dir / f'{basename}_fire_occurrence.tif'
        with rasterio.open(
            occ_path, 'w', driver='GTiff',
            height=shape[0], width=shape[1], count=1, dtype=occ_map.dtype
        ) as dst:
            dst.write(occ_map, 1)
        
        prob_map = fire_proba.reshape(shape)
        prob_path = output_dir / f'{basename}_fire_probability.tif'
        with rasterio.open(
            prob_path, 'w', driver='GTiff',
            height=shape[0], width=shape[1], count=1, dtype=prob_map.dtype
        ) as dst:
            dst.write(prob_map, 1)
        
        return occ_path, prob_path
    
    def save_predictions_csv(self, predictions, output_dir, basename='predictions'):
        """Save predictions as CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        y_pred, fire_proba = predictions
        df = pd.DataFrame({
            'pixel_id': np.arange(len(y_pred)),
            'fire_occurrence': y_pred,
            'fire_probability': fire_proba
        })
        
        csv_path = output_dir / f'{basename}_predictions.csv'
        df.to_csv(csv_path, index=False)
        return csv_path


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Generate wildfire predictions on new GeoTIFF rasters')
    parser.add_argument('--input_dir', type=str, default='data/new_rasters', help='Directory containing input GeoTIFF files')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions', help='Directory to save prediction outputs')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing trained models')
    parser.add_argument('--reference_shape', type=int, nargs=2, default=[244, 344], help='Reference shape for raster alignment (rows cols)')
    
    args = parser.parse_args()
    
    predictor = WildfirePredictor(models_dir=args.models_dir)
    
    try:
        X_raw, shape = predictor.load_and_align_rasters(args.input_dir, reference_shape=tuple(args.reference_shape))
        print(f"Loaded {X_raw.shape[1]} features from {args.input_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"Loaded rasters with shape {shape}. Total pixels: {X_raw.shape[0]}")
    
    X = predictor.prepare_features(X_raw)
    print(f"Prepared features: {X.shape[1]} engineered features")
    
    y_pred, fire_proba = predictor.predict_fire_occurrence(X)
    print(f"Fire occurrence predictions: {y_pred.sum()} fire pixels out of {len(y_pred)}")
    
    fire_mask = y_pred == 1
    y_magnitude = predictor.predict_fire_magnitude(X, fire_mask)
    
    occ_path, prob_path = predictor.create_output_rasters((y_pred, fire_proba), shape, args.output_dir)
    csv_path = predictor.save_predictions_csv((y_pred, fire_proba), args.output_dir)
    
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - {occ_path.name}")
    print(f"  - {prob_path.name}")
    print(f"  - {csv_path.name}")


if __name__ == '__main__':
    main()
