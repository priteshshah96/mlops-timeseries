from pathlib import Path
import pandas as pd
import numpy as np
import gc
import logging
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage of DataFrame by optimizing dtypes"""
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            if str(df[col].dtype).startswith('int'):
                col_min, col_max = df[col].min(), df[col].max()
                if col_min >= 0:
                    if col_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif col_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    else:
                        df[col] = df[col].astype(np.uint32)
            elif str(df[col].dtype).startswith('float'):
                df[col] = df[col].astype(np.float32)
    return df

def check_data_quality(df: pd.DataFrame) -> None:
    """Check data quality and log issues"""
    logger.info("Checking data quality...")
    
    # Check missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    if missing.any():
        logger.warning("Missing values found:")
        for col in missing[missing > 0].index:
            logger.warning(f"- {col}: {missing[col]} ({missing_pct[col]:.2f}%)")

def preprocess_comprehensive(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive preprocessing for electricity demand data"""
    initial_memory = df.memory_usage().sum() / 1024**2
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    initial_rows = len(df)
    
    # 1. Initial data quality check
    check_data_quality(df)
    
    # 2. Keep only essential columns
    essential_weather_cols = [
        'temperature_2m', 'relative_humidity_2m', 
        'wind_speed_10m', 'precipitation', 'cloud_cover'
    ]
    
    keep_cols = [
        'unique_id', 'timestamp', 'y', 'building_id', 'building_class',
        'location_id', 'cluster_size'
    ] + [col for col in essential_weather_cols if col in df.columns]
    
    df = df[keep_cols]
    gc.collect()
    logger.info(f"Kept {len(keep_cols)} essential columns")
    
    # 3. Handle missing values in weather data
    logger.info("Handling missing weather data...")
    weather_cols = [col for col in essential_weather_cols if col in df.columns]
    
    # Forward fill then backward fill within each location
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby('location_id')[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )
    
    # 4. Create time features
    logger.info("Creating time features...")
    df['hour'] = df['timestamp'].dt.hour.astype(np.uint8)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype(np.uint8)
    df['month'] = df['timestamp'].dt.month.astype(np.uint8)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.uint8)
    
    # 5. Create lag features
    logger.info("Creating lag features...")
    # Previous day and week
    for lag in [24, 168]:  # 24 hours and 1 week
        df[f'demand_lag_{lag}h'] = df.groupby('unique_id')['y'].shift(lag)
    
    # Rolling mean for demand
    df['demand_rolling_mean_24h'] = (
        df.groupby('unique_id')['y']
        .rolling(window=24, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    gc.collect()
    
    # 6. Building features
    logger.info("Creating building features...")
    # Encode building class
    le = LabelEncoder()
    df['building_class_encoded'] = le.fit_transform(df['building_class'])
    df.drop('building_class', axis=1, inplace=True)
    
    # Normalize by cluster size
    df['consumption_per_unit'] = df['y'] / df['cluster_size']
    
    # 7. Handle outliers in target variable
    logger.info("Handling outliers...")
    def cap_outliers(group):
        q1 = group['y'].quantile(0.25)
        q3 = group['y'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        group['y'] = np.clip(group['y'], lower_bound, upper_bound)
        return group
    
    df = df.groupby('unique_id', group_keys=False).apply(cap_outliers)
    
    # 8. Final cleanup
    logger.info("Final cleanup...")
    # Fill any remaining NaN values in lag features with 0
    lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    # Optimize memory usage
    df = reduce_memory_usage(df)
    gc.collect()
    
    final_memory = df.memory_usage().sum() / 1024**2
    logger.info(f"Final memory usage: {final_memory:.2f} MB")
    logger.info(f"Memory reduction: {((initial_memory - final_memory) / initial_memory) * 100:.1f}%")
    logger.info(f"Rows: {initial_rows} -> {len(df)} ({len(df)/initial_rows*100:.1f}% kept)")
    
    return df

def main():
    logger.info("Starting preprocessing...")
    
    # Set up paths
    data_dir = Path('C:/Users/prite/Desktop/mlops-timeseries/data')
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    input_path = processed_dir / 'processed_data.parquet'
    output_path = processed_dir / 'processed_data2.parquet'
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    
    # Process data
    processed_df = preprocess_comprehensive(df)
    
    # Save processed data
    logger.info(f"Saving processed data to {output_path}")
    processed_df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    logger.info(f"Final shape: {processed_df.shape}")
    logger.info("Features created:")
    for col in processed_df.columns:
        logger.info(f"- {col}: {processed_df[col].dtype}")

if __name__ == "__main__":
    main()