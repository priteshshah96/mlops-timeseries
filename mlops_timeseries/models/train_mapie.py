import numpy as np
import xgboost as xgb
from mapie.regression import MapieRegressor
from mapie.conformity_scores import AbsoluteConformityScore
from tqdm import tqdm
import logging
import traceback
from pathlib import Path
import time
import gc
import joblib
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomConformityScore(AbsoluteConformityScore):
    """Custom conformity score with dynamic epsilon scaling"""
    def __init__(self, y_val, y_pred_val):
        super().__init__()
        pred_scale = np.abs(y_pred_val).mean()
        self.eps = max(pred_scale * 1e-3, 1e-3)

    def get_conformity_scores(self, y, y_pred, **kwargs):
        scores = np.abs(y - y_pred)
        scores = np.maximum(scores, self.eps)
        return scores

def train_xgboost_with_mapie(train_df, valid_df, features, target, best_params, save_path="C:/Users/prite/Desktop/mlops-timeseries/data/checkpoints/"):
    """Train XGBoost model with MAPIE uncertainty estimation"""
    try:
        Path(save_path).mkdir(parents=True, exist_ok=True)  # Create save directory if it doesn't exist

        # Convert parameters for scikit-learn API
        sklearn_params = {
            'objective': 'reg:squarederror',
            'learning_rate': best_params.get('eta', 0.1),
            'max_depth': best_params.get('max_depth', 6),
            'min_child_weight': best_params.get('min_child_weight', 1),
            'subsample': best_params.get('subsample', 0.8),
            'colsample_bytree': best_params.get('colsample_bytree', 0.8),
            'reg_alpha': best_params.get('alpha', 0.001),
            'reg_lambda': best_params.get('lambda', 0.001),
            'tree_method': 'hist',
            'n_estimators': 1000,
            'verbosity': 0,
            'early_stopping_rounds': 50
        }
        
        # Train base model
        logger.info("Training base XGBoost model...")
        model = xgb.XGBRegressor(**sklearn_params)
        
        with tqdm(total=1, desc="Training XGBoost") as pbar:
            model.fit(
                train_df[features],
                train_df[target],
                eval_set=[(valid_df[features], valid_df[target])],
                verbose=False
            )
            pbar.update(1)

        # Save base XGBoost model
        logger.info("Saving base XGBoost model...")
        joblib.dump(model, f'{save_path}/xgboost_base.pkl')
        
        # Get validation predictions
        logger.info("Computing validation predictions...")
        val_preds = model.predict(valid_df[features])
        
        # Initialize conformity score
        logger.info("Initializing conformity score...")
        conformity_score = CustomConformityScore(
            valid_df[target].values,
            val_preds
        )
        
        # Configure and train MAPIE
        logger.info("Training MAPIE...")
        mapie_model = MapieRegressor(
            model,
            method="plus",
            cv="prefit",
            conformity_score=conformity_score,
            n_jobs=1,
            random_state=42
        )
        
        with tqdm(total=1, desc="Training MAPIE") as pbar:
            mapie_model.fit(
                valid_df[features],
                valid_df[target]
            )
            pbar.update(1)

        # Save MAPIE model as the final model
        logger.info("Saving final MAPIE model (XGBoost with uncertainty)...")
        joblib.dump(mapie_model, f'{save_path}/xgboost_final.pkl')
        
        logger.info("Models saved successfully")
        
        # Validate prediction intervals with debugging
        logger.info("Validating prediction intervals...")
        sample_data = valid_df[features][:100]
        y_pred, y_pis = mapie_model.predict(sample_data, alpha=[0.1])
        
        # Debug prediction shapes
        logger.info(f"Prediction shape: {y_pred.shape}")
        logger.info(f"Intervals shape: {y_pis.shape}")
        
        # Get actual values
        y_true = valid_df[target].iloc[:100].values
        
        # Calculate intervals correctly based on MAPIE output shape
        lower_bound = y_pis[:, 0, 0]  # First interval bound
        upper_bound = y_pis[:, 1, 0]  # Second interval bound
        
        # Calculate metrics
        coverage = np.mean(
            (y_true >= lower_bound) & (y_true <= upper_bound)
        )
        interval_width = np.mean(upper_bound - lower_bound)
        
        logger.info(f"Validation coverage: {coverage:.4f}")
        logger.info(f"Average interval width: {interval_width:.4f}")
        
        # Log some example predictions
        logger.info("\nExample predictions (first 5):")
        for i in range(5):
            logger.info(
                f"True: {y_true[i]:.2f}, "
                f"Pred: {y_pred[i]:.2f}, "
                f"Interval: [{lower_bound[i]:.2f}, {upper_bound[i]:.2f}]"
            )
        
        return model, mapie_model
            
    except Exception as e:
        logger.error(f"Error in XGBoost+MAPIE training: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        # Load data
        data_path = Path('C:/Users/prite/Desktop/mlops-timeseries/data/processed/processed_data2.parquet')
        df = pd.read_parquet(data_path)
        
        # Define features
        features = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
            'building_class_encoded', 'cluster_size',
            'demand_lag_24h', 'demand_lag_168h', 'demand_rolling_mean_24h'
        ]
        target = 'y'
        
        # Create splits
        train_end = '2014-07-01'
        valid_end = '2014-10-01'
        
        train_df = df[df['timestamp'] < train_end]
        valid_df = df[(df['timestamp'] >= train_end) & 
                     (df['timestamp'] < valid_end)]
        
        # Best known XGBoost parameters
        best_params = {
            'max_depth': 7,
            'eta': 0.055258462001045956,
            'min_child_weight': 4,
            'subsample': 0.8893422718168111,
            'colsample_bytree': 0.7590424408711858,
            'alpha': 7.731717010781557e-05,
            'lambda': 1.8191747105416762
        }
        
        # Train models
        model, mapie_model = train_xgboost_with_mapie(
            train_df, valid_df, features, target, best_params
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        gc.collect()
