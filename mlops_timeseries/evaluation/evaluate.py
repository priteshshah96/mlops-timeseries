# mlops_timeseries/evaluation/evaluate.py

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from datetime import datetime
import gc
import warnings
from mapie.regression import MapieRegressor
# Suppress warnings
warnings.filterwarnings('ignore')

from mapie.conformity_scores import AbsoluteConformityScore
import numpy as np

class CustomConformityScore(AbsoluteConformityScore):
    def __init__(self, y_val, y_pred_val):
        super().__init__()
        pred_scale = np.abs(y_pred_val).mean()
        self.eps = max(pred_scale * 1e-3, 1e-3)

    def get_conformity_scores(self, y, y_pred, **kwargs):
        scores = np.abs(y - y_pred)
        scores = np.maximum(scores, self.eps)
        return scores

class ModelEvaluator:
    """Evaluates trained models and generates performance reports"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.models_dir = data_dir / 'models'
        self.checkpoints_dir = data_dir / 'checkpoints'
        self.results_dir = data_dir / 'results'
        self.plots_dir = self.results_dir / 'plots'
        
        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.checkpoints_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logger()
        
        # Load data splits
        self._load_data_splits()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / 'evaluation.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def _load_data_splits(self):
        """Load data splits from checkpoint"""
        try:
            # Load processed data
            data = pd.read_parquet(self.data_dir / 'processed/processed_data2.parquet')
            
            # Define features
            self.features = [
                'hour', 'day_of_week', 'month', 'is_weekend',
                'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                'building_class_encoded', 'cluster_size',
                'demand_lag_24h', 'demand_lag_168h', 'demand_rolling_mean_24h'
            ]
            
            # Create splits
            train_end = '2014-07-01'
            valid_end = '2014-10-01'
            
            self.train_df = data[data['timestamp'] < train_end]
            self.valid_df = data[(data['timestamp'] >= train_end) & 
                               (data['timestamp'] < valid_end)]
            self.test_df = data[data['timestamp'] >= valid_end]
            
            self.logger.info(f"Loaded data splits - Test set shape: {self.test_df.shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading data splits: {str(e)}")
            raise
    
    def load_models(self) -> Dict:
        """Load trained models from checkpoints"""
        models = {}
        try:
            # Define model paths
            model_files = {
                'lightgbm': self.checkpoints_dir / 'lightgbm_final.pkl',
                'xgboost': self.checkpoints_dir / 'xgboost_final.pkl'
            }
            
            # Load each model if it exists
            for model_name, model_path in model_files.items():
                if model_path.exists():
                    self.logger.info(f"Loading {model_name} from {model_path}")
                    models[model_name] = joblib.load(model_path)
                    self.logger.info(f"Successfully loaded {model_name} model")
                else:
                    self.logger.warning(f"Model file not found: {model_path}")
            
            if not models:
                raise FileNotFoundError("No model files found in checkpoints directory")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_intervals: np.ndarray, model_name: str,
                        timestamp: pd.Series, num_samples: int = 1000):
        """Create prediction vs actual plots"""
        plt.figure(figsize=(15, 8))
        
        # Sample points for clarity
        idx = np.random.choice(len(y_true), num_samples, replace=False)
        
        # Sort by timestamp
        sort_idx = np.argsort(timestamp.iloc[idx])
        timestamps = timestamp.iloc[idx].iloc[sort_idx]
        y_true_plot = y_true[idx][sort_idx]
        y_pred_plot = y_pred[idx][sort_idx]
        
        # Get interval bounds
        lower_bound = y_pred_intervals[idx][sort_idx][:, 0, 0]
        upper_bound = y_pred_intervals[idx][sort_idx][:, 1, 0]
        
        # Plot predictions and intervals
        plt.plot(timestamps, y_true_plot, label='Actual', alpha=0.7)
        plt.plot(timestamps, y_pred_plot, label='Predicted', alpha=0.7)
        plt.fill_between(timestamps, 
                        lower_bound,
                        upper_bound,
                        alpha=0.2, label='90% Prediction Interval')
        
        plt.title(f'{model_name} Predictions vs Actual')
        plt.xlabel('Timestamp')
        plt.ylabel('Demand')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.plots_dir / f'{model_name.lower()}_predictions.png')
        plt.close()
    
    def generate_summary_plots(self, model_name: str, y_true: np.ndarray, 
                             y_pred: np.ndarray, y_pred_intervals: np.ndarray):
        """Generate comprehensive summary plots"""
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Residual Plot
        residuals = y_true - y_pred
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residual Plot')
        
        # 2. Prediction vs Actual
        axes[0, 1].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title('Prediction vs Actual')
        
        # 3. Residual Distribution
        sns.histplot(residuals, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].set_xlabel('Residual Value')
        
        # 4. Interval Width Distribution
        interval_widths = y_pred_intervals[:, 1, 0] - y_pred_intervals[:, 0, 0]
        sns.histplot(interval_widths, kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Prediction Interval Width Distribution')
        axes[1, 1].set_xlabel('Interval Width')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{model_name.lower()}_summary_plots.png')
        plt.close()
    
    def evaluate_all_models(self):
        """Evaluate all trained models with improved metrics and plots"""
        try:
            models = self.load_models()
            
            results = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for model_name, model in models.items():
                self.logger.info(f"\nEvaluating {model_name}...")
                
                # Handle the case where the model is stored as a dictionary
                if isinstance(model, dict):
                    # Assume the actual model is stored under a key like 'model' or 'mapie_model'
                    base_model = model.get('model') or model.get('mapie_model') or model
                    mapie_model = model.get('mapie_model', None)
                else:
                    # If not a dict, assume it’s directly a model
                    base_model = model
                    mapie_model = model if isinstance(model, MapieRegressor) else None
                
                # Check if base_model is valid
                if not hasattr(base_model, "predict"):
                    self.logger.error(f"Loaded {model_name} is not a valid model with a predict method.")
                    continue
                
                # Make predictions
                y_pred = base_model.predict(self.test_df[self.features])
                
                if mapie_model and isinstance(mapie_model, MapieRegressor):
                    _, y_pred_intervals = mapie_model.predict(
                        self.test_df[self.features],
                        alpha=[0.1]
                    )
                else:
                    y_pred_intervals = None

                # Calculate metrics
                metrics = self.evaluate_predictions(
                    self.test_df['y'].values,
                    y_pred,
                    y_pred_intervals
                )
                
                # Convert numpy types to Python native types for JSON serialization
                metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in metrics.items()}
                
                # Generate plots
                self.plot_predictions(
                    self.test_df['y'].values,
                    y_pred,
                    y_pred_intervals,
                    model_name,
                    self.test_df['timestamp']
                )
                
                # Generate summary plots
                self.generate_summary_plots(
                    model_name,
                    self.test_df['y'].values,
                    y_pred,
                    y_pred_intervals
                )
                
                # Log results
                self.logger.info(f"\n{model_name} Results:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        self.logger.info(f"{metric}: {value:.4f}")
                    else:
                        self.logger.info(f"{metric}: {value}")
                
                results[model_name] = metrics
            
            # Save results
            results_file = self.results_dir / f'evaluation_results_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            self.logger.info(f"\nEvaluation completed. Results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            raise
        finally:
            gc.collect()



    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_intervals: np.ndarray = None) -> Dict:
        """Calculate evaluation metrics with improved error handling"""
        try:
            # Basic metrics
            metrics = {
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2': float(r2_score(y_true, y_pred))
            }
            
            # Calculate MAPE safely
            mask = y_true != 0
            if mask.any():
                mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
                metrics['mape'] = mape
            else:
                metrics['mape'] = None
            
            # Additional metrics
            metrics['mean_actual'] = float(np.mean(y_true))
            metrics['mean_predicted'] = float(np.mean(y_pred))
            metrics['std_actual'] = float(np.std(y_true))
            metrics['std_predicted'] = float(np.std(y_pred))
            
            # Percentage of accurate predictions within different thresholds
            for threshold in [5, 10, 20]:
                percent_error = np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8)) * 100
                metrics[f'within_{threshold}p_error'] = float(np.mean(percent_error <= threshold) * 100)
            
            if y_pred_intervals is not None:
                lower_bound = y_pred_intervals[:, 0, 0]
                upper_bound = y_pred_intervals[:, 1, 0]
                
                # Basic interval metrics
                coverage = float(np.mean((y_true >= lower_bound) & (y_true <= upper_bound)))
                interval_width = float(np.mean(upper_bound - lower_bound))
                
                # Additional interval metrics
                metrics.update({
                    'prediction_interval_coverage': coverage,
                    'target_coverage': 0.9,  # Since alpha=0.1
                    'average_interval_width': interval_width,
                    'relative_interval_width': float(interval_width / np.mean(np.abs(y_true))),
                    'min_interval_width': float(np.min(upper_bound - lower_bound)),
                    'max_interval_width': float(np.max(upper_bound - lower_bound))
                })
                
                metrics['interval_sharpness'] = float(np.mean(upper_bound - lower_bound))
                interval_centers = (upper_bound + lower_bound) / 2
                metrics['avg_deviation_from_center'] = float(np.mean(np.abs(y_true - interval_centers)))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
def main():
    """Main function to run evaluation"""
    try:
        data_dir = Path('C:/Users/prite/Desktop/mlops-timeseries/data')
        
        evaluator = ModelEvaluator(data_dir)
        evaluator.evaluate_all_models()
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()