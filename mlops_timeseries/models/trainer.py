# mlops_timeseries/models/trainer.py

import os
import io
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mapie.regression import MapieRegressor
from mapie.conformity_scores import AbsoluteConformityScore
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
from typing import Dict, List, Tuple, Optional
import joblib
import json
from datetime import datetime
import warnings
import traceback
import psutil
import time
from contextlib import redirect_stdout
from tqdm.auto import tqdm
import gc

# Suppress warnings
warnings.filterwarnings('ignore')


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging with file rotation for extended runs."""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("ModelTrainer")
    logger.setLevel(logging.DEBUG)

    # File handler for all logs with rotation
    log_file = log_dir / f"training_{timestamp}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler for INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Error log handler
    error_file = log_dir / f"errors_{timestamp}.log"
    eh = logging.FileHandler(error_file)
    eh.setLevel(logging.ERROR)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')

    fh.setFormatter(detailed_formatter)
    ch.setFormatter(simple_formatter)
    eh.setFormatter(detailed_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.addHandler(eh)

    return logger

class ExperimentTracker:
    """Track and save experiment results."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_file = results_dir / "experiment_results.json"
        self.current_experiment = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "models": {},
            "system_info": self._get_system_info()
        }

    def _get_system_info(self) -> Dict:
        return {
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "platform": sys.platform
        }

    def log_model_results(self,
                          model_name: str,
                          metrics: Dict,
                          params: Dict,
                          training_time: float,
                          memory_usage: float):
        self.current_experiment["models"][model_name] = {
            "metrics": metrics,
            "parameters": params,
            "training_time": training_time,
            "memory_usage": memory_usage
        }

    def save_results(self):
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                all_experiments = json.load(f)
        else:
            all_experiments = []

        all_experiments.append(self.current_experiment)

        with open(self.results_file, 'w') as f:
            json.dump(all_experiments, f, indent=2)

class ModelCheckpoint:
    """Handle model checkpointing."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, state: Dict, filename: str):
        checkpoint_path = self.checkpoint_dir / filename
        joblib.dump(state, checkpoint_path)

    def load_checkpoint(self, filename: str) -> Optional[Dict]:
        checkpoint_path = self.checkpoint_dir / filename
        if checkpoint_path.exists():
            return joblib.load(checkpoint_path)
        return None

class ModelTrainer:
    """Main class for model training and evaluation."""

    def __init__(self, data_dir: Path, n_trials: int = 100, n_jobs: int = -1):
        self.data_dir = data_dir
        self.n_trials = n_trials
        self.n_jobs = n_jobs if n_jobs > 0 else psutil.cpu_count() - 1

        # Set up directories
        self.models_dir = data_dir / 'models'
        self.results_dir = data_dir / 'results'
        self.log_dir = data_dir / 'logs'
        self.checkpoint_dir = data_dir / 'checkpoints'

        for directory in [self.models_dir, self.results_dir, self.log_dir, self.checkpoint_dir]:
            directory.mkdir(exist_ok=True)

        # Initialize components
        self.logger = setup_logging(self.log_dir)
        self.experiment_tracker = ExperimentTracker(self.results_dir)
        self.checkpoint_manager = ModelCheckpoint(self.checkpoint_dir)

        # Storage
        self.features = None
        self.target = 'y'
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.models = {}
        self.mapie_models = {}
        self.results = {}

        self.logger.info(f"Initialized ModelTrainer with {self.n_trials} trials and {self.n_jobs} workers")

    def load_and_prepare_data(self) -> None:
        """Load and prepare data with validations."""
        try:
            self.logger.info("Loading data...")
            start_time = time.time()

            # Ensure the processed data file exists
            processed_data_path = self.data_dir / 'processed/processed_data2.parquet'
            if not processed_data_path.exists():
                raise FileNotFoundError(f"Processed data file not found: {processed_data_path}")

            # Load data
            df = pd.read_parquet(processed_data_path)

            # Define features
            self.features = [
                'hour', 'day_of_week', 'month', 'is_weekend',
                'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                'building_class_encoded', 'cluster_size',
                'demand_lag_24h', 'demand_lag_168h', 'demand_rolling_mean_24h'
            ]

            # Verify features
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                self.features = [f for f in self.features if f in df.columns]

            if not self.features:
                raise ValueError("No valid features available in the dataset.")

            # Create time splits
            train_end = '2014-07-01'
            valid_end = '2014-10-01'

            self.train_df = df[df['timestamp'] < train_end]
            self.valid_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < valid_end)]
            self.test_df = df[df['timestamp'] >= valid_end]

            if self.valid_df.empty:
                raise ValueError("Validation dataset is empty after splitting.")

            # Log data info
            self.logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Train shape: {self.train_df.shape}")
            self.logger.info(f"Valid shape: {self.valid_df.shape}")
            self.logger.info(f"Test shape: {self.test_df.shape}")
            self.logger.info(f"Features: {self.features}")

            # Save data splits checkpoint
            self.checkpoint_manager.save_checkpoint(
                {
                    'features': self.features,
                    'train_index': self.train_df.index,
                    'valid_index': self.valid_df.index,
                    'test_index': self.test_df.index
                },
                'data_splits.pkl'
            )

        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def optimize_lightgbm(self, trial: optuna.Trial) -> float:
        """Optimize LightGBM hyperparameters"""
        start_time = time.time()
        try:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            
            # Create model
            model = lgb.LGBMRegressor(**params)
            
            # Simple cross-validation to get a more robust score
            scores = []
            for i in range(3):  # 3-fold time-based validation
                fold_size = len(self.train_df) // 3
                val_start = i * fold_size
                val_end = (i + 1) * fold_size
                
                train_idx = list(range(0, val_start)) + list(range(val_end, len(self.train_df)))
                val_idx = list(range(val_start, val_end))
                
                model.fit(
                    self.train_df.iloc[train_idx][self.features],
                    self.train_df.iloc[train_idx][self.target],
                    eval_set=[(self.train_df.iloc[val_idx][self.features], 
                            self.train_df.iloc[val_idx][self.target])],
                    callbacks=[lgb.early_stopping(stopping_rounds=50),
                            lgb.log_evaluation(0)]
                )
                
                pred = model.predict(self.train_df.iloc[val_idx][self.features])
                score = mean_absolute_error(self.train_df.iloc[val_idx][self.target], pred)
                scores.append(score)
            
            final_score = np.mean(scores)
            
            self.logger.debug(
                f"LightGBM trial {trial.number}: score={final_score:.4f}, "
                f"time={time.time()-start_time:.2f}s, params={params}"
            )
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error in LightGBM trial {trial.number}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise optuna.TrialPruned()

    
    def optimize_xgboost(self, trial: optuna.Trial) -> float:
        """Optimize XGBoost hyperparameters with proper callback implementation"""
        start_time = time.time()
        trial_id = trial.number
        try:
            self.logger.info(f"Starting trial {trial_id}")
            
            params = {
                'objective': 'reg:squarederror',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
                'tree_method': 'hist',
                'verbosity': 0,
                'nthread': 1
            }
            
            dtrain = xgb.DMatrix(
                self.train_df[self.features],
                label=self.train_df[self.target],
                nthread=1
            )
            dvalid = xgb.DMatrix(
                self.valid_df[self.features],
                label=self.valid_df[self.target],
                nthread=1
            )
            
            evals = [(dtrain, 'train'), (dvalid, 'validation')]
            evals_result = {}

            class CustomCallback(xgb.callback.TrainingCallback):
                def __init__(self, trial_id, logger):
                    self.trial_id = trial_id
                    self.logger = logger
                    self.best_score = float('inf')
                    self.stopping_rounds = 0
                    self.stopping_threshold = 50

                def after_iteration(self, model, epoch, evals_log):
                    score = evals_log['validation']['rmse'][-1]
                    
                    if score < self.best_score:
                        self.best_score = score
                        self.stopping_rounds = 0
                    else:
                        self.stopping_rounds += 1

                    if epoch % 100 == 0:
                        self.logger.info(
                            f"Trial {self.trial_id} - Iteration {epoch}: "
                            f"validation score = {score:.4f}"
                        )

                    if self.stopping_rounds >= self.stopping_threshold:
                        self.logger.info(
                            f"Trial {self.trial_id} stopping early at iteration {epoch}"
                        )
                        return True
                    
                    return False

            self.logger.info(f"Training XGBoost model for trial {trial_id}")
            
            # Create callback instance
            custom_callback = CustomCallback(trial_id, self.logger)
            
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=evals,
                evals_result=evals_result,
                callbacks=[custom_callback],
                verbose_eval=False
            )
            
            final_score = min(evals_result['validation']['rmse'])
            
            self.logger.info(
                f"Trial {trial_id} completed: "
                f"score={final_score:.4f}, "
                f"time={time.time()-start_time:.2f}s"
            )
            
            # Clean up
            del dtrain, dvalid, bst, evals_result
            gc.collect()
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
        raise optuna.TrialPruned()

    class CustomConformityScore(AbsoluteConformityScore):
        """Custom conformity score with dynamic epsilon scaling"""
        def __init__(self, y_val, y_pred_val):
            # Scale epsilon based on the prediction scale
            pred_scale = np.abs(y_pred_val).mean()
            eps = max(pred_scale * 1e-6, 1e-6)  # Ensure minimum epsilon
            super().__init__(eps=eps)

        def get_conformity_scores(self, y, y_pred, **kwargs):
            """Calculate conformity scores with additional validation"""
            scores = np.abs(y - y_pred)
            
            # Ensure scores are not too small
            scores = np.maximum(scores, self.eps)
            
            return scores

    def train_model_with_mapie(self, model_type: str, best_params: Dict) -> None:
        """Train final model with MAPIE using improved conformity score"""
        start_time = time.time()
        try:
            self.logger.info(f"Training final {model_type} model with MAPIE...")

            if model_type == "xgboost":
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
                self.logger.info("Training base model...")
                model = xgb.XGBRegressor(**sklearn_params)
                
                with tqdm(total=1, desc=f"Training {model_type}") as pbar:
                    model.fit(
                        self.train_df[self.features],
                        self.train_df[self.target],
                        eval_set=[(self.valid_df[self.features], self.valid_df[self.target])],
                        verbose=False
                    )
                    pbar.update(1)

                # Get validation predictions for conformity score initialization
                val_preds = model.predict(self.valid_df[self.features])
                
                # Initialize custom conformity score with validation data
                conformity_score = CustomConformityScore(
                    self.valid_df[self.target].values,
                    val_preds
                )
                
                # Configure MAPIE with custom conformity score and more robust settings
                self.logger.info("Training MAPIE...")
                mapie_model = MapieRegressor(
                    model,
                    method="plus",
                    cv="prefit",
                    conformity_score=conformity_score,
                    n_jobs=1,
                    random_state=42
                )
                
                # Train MAPIE with progress monitoring and error handling
                try:
                    with tqdm(total=1, desc="Training MAPIE") as pbar:
                        mapie_model.fit(
                            self.valid_df[self.features],
                            self.valid_df[self.target]
                        )
                        pbar.update(1)
                    self.logger.info("MAPIE training completed successfully")
                    
                    # Validate prediction intervals
                    _, y_pis = mapie_model.predict(
                        self.valid_df[self.features][:100],
                        alpha=0.1
                    )
                    
                    # Calculate and log coverage metrics
                    coverage = np.mean(
                        (self.valid_df[self.target].iloc[:100].values >= y_pis[:, 0, 0]) & 
                        (self.valid_df[self.target].iloc[:100].values <= y_pis[:, 0, 1])
                    )
                    interval_width = np.mean(y_pis[:, 0, 1] - y_pis[:, 0, 0])
                    
                    self.logger.info(f"Validation coverage: {coverage:.4f}")
                    self.logger.info(f"Average interval width: {interval_width:.4f}")
                    
                    # Verify intervals are reasonable
                    if interval_width <= 0 or coverage < 0.01:
                        raise ValueError(
                            f"Invalid prediction intervals: width={interval_width:.4f}, "
                            f"coverage={coverage:.4f}"
                        )
                    
                except Exception as e:
                    self.logger.error(f"MAPIE training failed: {str(e)}")
                    raise
                
                # Save models
                self.models[model_type] = model
                self.mapie_models[model_type] = mapie_model
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    {
                        "model": model,
                        "mapie_model": mapie_model,
                        "params": sklearn_params,
                        "conformity_score": conformity_score
                    },
                    f"{model_type}_final.pkl"
                )
                
                self.logger.info(
                    f"{model_type} training completed in {time.time()-start_time:.2f}s"
                )

        except Exception as e:
            self.logger.error(f"Error in {model_type} training: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        
    def _get_checkpoint_state(self) -> Optional[Dict]:
        """Get the latest checkpoint state"""
        try:
            checkpoint = self.checkpoint_manager.load_checkpoint('pipeline_state.pkl')
            if checkpoint:
                self.logger.info("Found existing checkpoint")
                return checkpoint
            return None
        except Exception as e:
            self.logger.warning(f"Error loading checkpoint: {str(e)}")
            return None

    def _save_checkpoint_state(self, state: Dict) -> None:
        """Save current pipeline state"""
        try:
            self.checkpoint_manager.save_checkpoint(state, 'pipeline_state.pkl')
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    # def train_models(self) -> None:
    #     """Train all models with optimization, checkpointing, and proper error handling"""
    #     try:
    #         # Memory cleanup before starting
    #         gc.collect()
            
    #         # Load checkpoint if exists
    #         checkpoint = self._get_checkpoint_state()
    #         models_to_train = ['lightgbm', 'xgboost']
            
    #         if checkpoint:
    #             completed_models = checkpoint.get('completed_models', [])
    #             models_to_train = [m for m in models_to_train if m not in completed_models]
                
    #             # Restore models and mapie models if they exist
    #             if 'models' in checkpoint:
    #                 self.models.update(checkpoint['models'])
    #             if 'mapie_models' in checkpoint:
    #                 self.mapie_models.update(checkpoint['mapie_models'])
                
    #             self.logger.info(f"Resuming training. Completed models: {completed_models}")
                
    #             # Cleanup checkpoint data
    #             del checkpoint
    #             gc.collect()
    #         else:
    #             completed_models = []
            
    #         for model_type in models_to_train:
    #             self.logger.info(f"\nOptimizing {model_type}...")
                
    #             # Clear memory before each model optimization
    #             gc.collect()
                
    #             study = optuna.create_study(
    #                 direction='minimize',
    #                 sampler=TPESampler(seed=42),
    #                 pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    #             )
                
    #             # Load study checkpoint if exists
    #             study_checkpoint = self.checkpoint_manager.load_checkpoint(f'{model_type}_study.pkl')
    #             if study_checkpoint:
    #                 try:
    #                     # Create a new storage with the saved trials
    #                     storage = optuna.storages.InMemoryStorage()
                        
    #                     # Add trials to the new storage
    #                     for trial_dict in study_checkpoint.get('trials', []):
    #                         if isinstance(trial_dict, optuna.Trial):
    #                             study._storage.create_new_trial(study._study_id, template_trial=trial_dict)
                        
    #                     self.logger.info(f"Restored {len(study.trials)} trials for {model_type}")
                        
    #                     # Cleanup checkpoint data
    #                     del study_checkpoint
    #                     gc.collect()
    #                 except Exception as e:
    #                     self.logger.warning(f"Failed to restore study checkpoint: {str(e)}. Starting fresh.")
                
    #             # Monitor memory usage
    #             current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    #             self.logger.info(f"Current memory usage: {current_memory:.2f} MB")
                
    #             # Set default parameters
    #             default_params = {
    #                 'lightgbm': {
    #                     'objective': 'regression',
    #                     'metric': 'mae',
    #                     'verbosity': -1,
    #                     'boosting_type': 'gbdt',
    #                     'num_leaves': 31,
    #                     'learning_rate': 0.1,
    #                     'feature_fraction': 0.8,
    #                     'bagging_fraction': 0.8,
    #                     'min_child_samples': 20,
    #                     'reg_alpha': 0.001,
    #                     'reg_lambda': 0.001
    #                 },
    #                 'xgboost': {
    #                     'objective': 'reg:squarederror',
    #                     'max_depth': 6,
    #                     'learning_rate': 0.1,
    #                     'n_estimators': 100,
    #                     'min_child_weight': 1,
    #                     'subsample': 0.8,
    #                     'colsample_bytree': 0.8,
    #                     'reg_alpha': 0.001,
    #                     'reg_lambda': 0.001,
    #                     'tree_method': 'hist'
    #                 }
    #             }
                
    #             try:
    #                 # Continue optimization from checkpoint
    #                 remaining_trials = self.n_trials - len(study.trials)
    #                 if remaining_trials > 0:
    #                     optimize_func = self.optimize_lightgbm if model_type == 'lightgbm' else self.optimize_xgboost
    #                     study.optimize(
    #                         optimize_func,
    #                         n_trials=remaining_trials,
    #                         n_jobs=self.n_jobs,
    #                         show_progress_bar=True
    #                     )
                        
    #                     # Save study checkpoint and cleanup
    #                     self.checkpoint_manager.save_checkpoint(
    #                         {'trials': study.trials},
    #                         f'{model_type}_study.pkl'
    #                     )
                        
    #                     # Clear memory after optimization
    #                     gc.collect()
                    
    #                 if study.best_trial is not None:
    #                     self.logger.info(f"Best {model_type} trial:")
    #                     self.logger.info(f"  Value: {study.best_value:.4f}")
    #                     self.logger.info(f"  Params: {study.best_params}")
    #                     best_params = study.best_params.copy()  # Create a copy to avoid reference issues
                        
    #                     # Clear study data
    #                     del study
    #                     gc.collect()
    #                 else:
    #                     self.logger.warning(f"No successful trials for {model_type}. Using default parameters.")
    #                     best_params = default_params[model_type].copy()
                    
    #                 # Train final model with best or default params
    #                 self.train_model_with_mapie(model_type, best_params)
                    
    #                 # Update checkpoint
    #                 completed_models.append(model_type)
    #                 self._save_checkpoint_state({
    #                     'completed_models': completed_models,
    #                     'models': self.models,
    #                     'mapie_models': self.mapie_models
    #                 })
                    
    #                 # Clear parameters
    #                 del best_params
    #                 gc.collect()
                    
    #             except Exception as e:
    #                 self.logger.error(f"Error during {model_type} optimization: {str(e)}")
    #                 self.logger.error(traceback.format_exc())
    #                 self.logger.warning(f"Falling back to default parameters for {model_type}")
                    
    #                 # Clear any remaining optimization data
    #                 if 'study' in locals():
    #                     del study
    #                 gc.collect()
                    
    #                 self.train_model_with_mapie(model_type, default_params[model_type].copy())
                    
    #                 # Update checkpoint even in case of failure
    #                 completed_models.append(model_type)
    #                 self._save_checkpoint_state({
    #                     'completed_models': completed_models,
    #                     'models': self.models,
    #                     'mapie_models': self.mapie_models
    #                 })
                
    #             # Monitor memory after each model
    #             current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    #             self.logger.info(f"Memory usage after {model_type}: {current_memory:.2f} MB")
                
    #     except Exception as e:
    #         self.logger.error("Error in model training pipeline")
    #         self.logger.error(traceback.format_exc())
    #         raise
    #     finally:
    #         # Final cleanup
    #         gc.collect()
    

    def train_models(self) -> None:
        """Train models using best known parameters"""
        try:
            # Memory cleanup before starting
            gc.collect()
            
            # Previous best XGBoost parameters from trial 85
            xgboost_best_params = {
                'max_depth': 7,
                'eta': 0.055258462001045956,
                'min_child_weight': 4,
                'subsample': 0.8893422718168111,
                'colsample_bytree': 0.7590424408711858,
                'alpha': 7.731717010781557e-05,
                'lambda': 1.8191747105416762
            }
            
            # Load checkpoint
            checkpoint = self._get_checkpoint_state()
            completed_models = checkpoint.get('completed_models', []) if checkpoint else []
            
            self.logger.info(f"Starting training with completed models: {completed_models}")
            
            # Restore existing models if any
            if checkpoint:
                if 'models' in checkpoint:
                    self.models.update(checkpoint['models'])
                if 'mapie_models' in checkpoint:
                    self.mapie_models.update(checkpoint['mapie_models'])
            
            # Train XGBoost if not already trained
            if 'xgboost' not in completed_models:
                self.logger.info("Training XGBoost with best parameters...")
                try:
                    self.train_model_with_mapie('xgboost', xgboost_best_params)
                    completed_models.append('xgboost')
                    
                    # Update checkpoint
                    self._save_checkpoint_state({
                        'completed_models': completed_models,
                        'models': self.models,
                        'mapie_models': self.mapie_models
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error training XGBoost: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    raise
            else:
                self.logger.info("XGBoost model already trained. Skipping...")
            
            self.logger.info("Model training completed")
            
        except Exception as e:
            self.logger.error("Error in model training pipeline")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            gc.collect()
    def evaluate_models(self) -> None:
        """Evaluate all trained models"""
        try:
            for model_type in self.models.keys():
                start_time = time.time()
                self.logger.info(f"\nEvaluating {model_type}...")
                
                model = self.models[model_type]
                mapie_model = self.mapie_models[model_type]
                
                # Make predictions
                predictions = model.predict(self.test_df[self.features])
                
                # Get prediction intervals
                y_pred, y_pis = mapie_model.predict(
                    self.test_df[self.features],
                    alpha=0.1
                )
                
                # Calculate metrics
                metrics = {
                    'mae': mean_absolute_error(self.test_df[self.target], predictions),
                    'rmse': np.sqrt(mean_squared_error(self.test_df[self.target], predictions)),
                    'r2': r2_score(self.test_df[self.target], predictions)
                }
                
                # Calculate coverage and interval width
                coverage = np.mean(
                    (self.test_df[self.target] >= y_pis[:, 0, 0]) & 
                    (self.test_df[self.target] <= y_pis[:, 0, 1])
                )
                interval_width = np.mean(y_pis[:, 0, 1] - y_pis[:, 0, 0])
                
                metrics['coverage'] = coverage
                metrics['interval_width'] = interval_width
                
                # Log metrics
                self.logger.info(f"{model_type} Results:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")
                
                # Save results
                self.results[model_type] = {
                    'metrics': metrics,
                    'predictions': predictions,
                    'intervals': y_pis,
                    'evaluation_time': time.time() - start_time
                }
                
                # Track experiment
                self.experiment_tracker.log_model_results(
                    model_type,
                    metrics,
                    model.get_params(),
                    time.time() - start_time,
                    psutil.Process().memory_info().rss / 1024 / 1024  # MB
                )
                
        except Exception as e:
            self.logger.error("Error in model evaluation")
            self.logger.error(traceback.format_exc())
            raise
    
    def save_results(self) -> None:
        """Save all results and models"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save models
            for model_type, model in self.models.items():
                joblib.dump(
                    model,
                    self.models_dir / f'{model_type}_{timestamp}.pkl'
                )
                joblib.dump(
                    self.mapie_models[model_type],
                    self.models_dir / f'{model_type}_mapie_{timestamp}.pkl'
                )
            
            # Save predictions and metrics
            results_df = pd.DataFrame(
                [
                    {
                        'model': name,
                        'timestamp': timestamp,
                        **results['metrics'],
                        'evaluation_time': results['evaluation_time']
                    }
                    for name, results in self.results.items()
                ]
            )
            
            results_df.to_csv(
                self.results_dir / f'metrics_{timestamp}.csv',
                index=False
            )
            
            # Save experiment results
            self.experiment_tracker.save_results()
            
            self.logger.info(f"Results saved with timestamp {timestamp}")
            
        except Exception as e:
            self.logger.error("Error saving results")
            self.logger.error(traceback.format_exc())
            raise

def main():
    try:
        data_dir = Path('C:/Users/prite/Desktop/mlops-timeseries/data')
        
        trainer = ModelTrainer(
            data_dir=data_dir,
            n_trials=100,  # High number of trials for overnight
            n_jobs=-1      # Use all cores
        )
        
        trainer.logger.info("Starting overnight training pipeline...")
        
        # Load data
        trainer.load_and_prepare_data()
        
        # Train models with checkpoint recovery
        trainer.train_models()
        
        # Evaluate and save only if not already done
        if not trainer.checkpoint_manager.load_checkpoint('evaluation_done.pkl'):
            trainer.evaluate_models()
            trainer.save_results()
            trainer.checkpoint_manager.save_checkpoint({'done': True}, 'evaluation_done.pkl')
        
        trainer.logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()