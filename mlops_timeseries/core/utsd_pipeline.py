# mlops_timeseries/core/utsd_pipeline.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class UTSDDomain(Enum):
    """UTSD dataset domains"""
    ENERGY = "energy"
    ENVIRONMENT = "environment"
    HEALTH = "health"
    IOT = "iot"
    NATURE = "nature"
    TRANSPORTATION = "transportation"
    WEB = "web"

class UTSDGeneration(Enum):
    """UTSD dataset generations"""
    G1 = "1G"
    G2 = "2G"
    G4 = "4G"
    G12 = "12G"

@dataclass
class UTSDConfig:
    """Configuration for UTSD processing"""
    domain: UTSDDomain
    generation: UTSDGeneration
    sequence_length: int
    forecast_horizon: int
    batch_size: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    scaling_method: str = "standard"
    random_seed: int = 42

class UTSDDataset(Dataset):
    """PyTorch Dataset for UTSD time series data"""
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        forecast_horizon: int,
        mode: str = "train"
    ):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[
            idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon
        ]
        return x, y

class UTSDPipeline:
    """Pipeline for processing UTSD data"""
    
    def __init__(self, config: UTSDConfig):
        self.config = config
        self.scalers = {}
        
    def load_data(self, data_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Load UTSD data for specified domain and generation
        
        Args:
            data_dir: Path to UTSD data directory
            
        Returns:
            Dictionary containing DataFrames for each time series in the domain
        """
        data_path = Path(data_dir) / self.config.generation.value / self.config.domain.value
        
        data_dict = {}
        for file_path in data_path.glob("*.csv"):
            series_name = file_path.stem
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            data_dict[series_name] = df
            
        return data_dict
    
    def preprocess_data(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, torch.utils.data.Dataset]]:
        """
        Preprocess UTSD data and create train/val/test splits
        
        Args:
            data_dict: Dictionary of DataFrames for each time series
            
        Returns:
            Dictionary containing train/val/test datasets for each time series
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        processed_data = {}
        
        for series_name, df in data_dict.items():
            # Select scaler based on config
            if self.config.scaling_method == "standard":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            # Scale the data
            data = df['value'].values.reshape(-1, 1)
            scaled_data = scaler.fit_transform(data)
            self.scalers[series_name] = scaler
            
            # Split the data
            n_samples = len(scaled_data)
            train_size = int(n_samples * self.config.train_ratio)
            val_size = int(n_samples * self.config.val_ratio)
            
            train_data = scaled_data[:train_size]
            val_data = scaled_data[train_size:train_size + val_size]
            test_data = scaled_data[train_size + val_size:]
            
            # Create datasets
            processed_data[series_name] = {
                'train': UTSDDataset(
                    train_data,
                    self.config.sequence_length,
                    self.config.forecast_horizon,
                    "train"
                ),
                'val': UTSDDataset(
                    val_data,
                    self.config.sequence_length,
                    self.config.forecast_horizon,
                    "val"
                ),
                'test': UTSDDataset(
                    test_data,
                    self.config.sequence_length,
                    self.config.forecast_horizon,
                    "test"
                )
            }
            
        return processed_data
    
    def create_dataloaders(
        self,
        datasets: Dict[str, Dict[str, torch.utils.data.Dataset]]
    ) -> Dict[str, Dict[str, DataLoader]]:
        """Create DataLoaders for each dataset"""
        dataloaders = {}
        
        for series_name, series_datasets in datasets.items():
            dataloaders[series_name] = {
                split: DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=(split == 'train')
                )
                for split, dataset in series_datasets.items()
            }
            
        return dataloaders
    
    def inverse_transform(
        self,
        predictions: torch.Tensor,
        series_name: str
    ) -> np.ndarray:
        """Convert scaled predictions back to original scale"""
        return self.scalers[series_name].inverse_transform(
            predictions.cpu().numpy()
        )

def create_utsd_pipeline(config_path: str) -> UTSDPipeline:
    """Create pipeline from config file"""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    # Convert string to enum values
    config_dict['domain'] = UTSDDomain(config_dict['domain'])
    config_dict['generation'] = UTSDGeneration(config_dict['generation'])
    
    config = UTSDConfig(**config_dict)
    return UTSDPipeline(config)