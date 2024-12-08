# mlops_timeseries/utils/preprocess_data.py
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import gc
import logging
import os
from datetime import datetime
import psutil
from tqdm import tqdm
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_project_paths():
    """Get the project paths"""
    # Get the current script directory (utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up two levels to reach mlops-timeseries (project root)
    # From: mlops_timeseries/utils/
    # To: mlops-timeseries/
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Define data directory
    data_dir = os.path.join(project_root, 'data')
    
    # Print paths for verification
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    
    # List files in data directory
    logger.info("Files in data directory:")
    try:
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"  - {file} ({size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"Error listing data directory: {str(e)}")
    
    return project_root, data_dir

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    logger.info(f"Current memory usage: {mem:.2f} MB")

def get_date_range_and_buildings(demand_path):
    """Get the actual date range and active buildings in the dataset"""
    # Read just the timestamp column to get the date range
    logger.info("Checking data availability...")
    
    try:
        # Read a small sample to get min/max dates
        sample_dates = pd.read_parquet(
            demand_path,
            columns=['timestamp', 'unique_id']
        )
        
        min_date = sample_dates['timestamp'].min()
        max_date = sample_dates['timestamp'].max()
        
        logger.info(f"Data available from {min_date} to {max_date}")
        
        # Get a sample of buildings that have data
        sample_buildings = sample_dates[
            (sample_dates['timestamp'] >= min_date) & 
            (sample_dates['timestamp'] <= min_date + pd.Timedelta(days=7))
        ]['unique_id'].unique()
        
        logger.info(f"Found {len(sample_buildings)} buildings with data in first week")
        
        return min_date, max_date, sample_buildings
        
    except Exception as e:
        logger.error(f"Error checking data availability: {traceback.format_exc()}")
        raise

class MinimalPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.demand_path = os.path.join(data_path, 'demand.parquet')
        self.weather_path = os.path.join(data_path, 'weather.parquet')
        self.metadata_path = os.path.join(data_path, 'metadata.parquet')
        
        # Verify data files exist
        self.verify_data_files()
        
        # Get date range and valid buildings
        self.start_date, self.end_date, self.valid_buildings = get_date_range_and_buildings(
            self.demand_path
        )
    
    def verify_data_files(self):
        """Verify that all required data files exist"""
        for path in [self.demand_path, self.weather_path, self.metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
            else:
                file_size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
                logger.info(f"Found data file: {path} ({file_size:.2f} MB)")

    def process_single_building(self, building_id, start_date, end_date):
        """Process a single building's data"""
        try:
            # Read demand data for the building
            demand_data = pd.read_parquet(
                self.demand_path,
                filters=[
                    ('unique_id', '=', building_id),
                    ('timestamp', '>=', start_date),
                    ('timestamp', '<', end_date)
                ]
            )

            if demand_data.empty:
                logger.warning(f"No data found for building {building_id}")
                return None

            # Read corresponding weather data
            weather_data = pd.read_parquet(
                self.weather_path,
                filters=[
                    ('timestamp', '>=', start_date),
                    ('timestamp', '<', end_date)
                ]
            )

            # Create features
            demand_data['hour'] = demand_data['timestamp'].dt.hour
            demand_data['day_of_week'] = demand_data['timestamp'].dt.dayofweek
            demand_data['month'] = demand_data['timestamp'].dt.month
            demand_data['is_weekend'] = demand_data['day_of_week'].isin([5, 6]).astype(int)
            
            # Merge with weather
            demand_data = pd.merge(demand_data, weather_data, on='timestamp', how='left')
            
            # Sort by timestamp for time-based features
            demand_data = demand_data.sort_values('timestamp')
            
            # Create lag features
            demand_data['demand_lag_24h'] = demand_data.groupby('unique_id')['y'].shift(24)
            demand_data['demand_lag_168h'] = demand_data.groupby('unique_id')['y'].shift(168)
            
            # Create rolling features
            demand_data['rolling_mean_24h'] = demand_data.groupby('unique_id')['y'].transform(
                lambda x: x.rolling(24, min_periods=1).mean()
            )
            
            return demand_data

        except Exception as e:
            logger.error(f"Error processing building {building_id}: {traceback.format_exc()}")
            return None

    def process_chunk(self, building_ids, start_date, end_date, output_dir):
        """Process a chunk of buildings"""
        processed_count = 0
        
        for building_id in building_ids:
            try:
                log_memory_usage()
                logger.info(f"Processing building {building_id}")
                
                # Process building
                building_data = self.process_single_building(building_id, start_date, end_date)
                
                if building_data is not None and not building_data.empty:
                    # Save individual building data
                    output_path = os.path.join(
                        output_dir, 
                        f"building_{building_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
                    )
                    building_data.to_parquet(output_path, engine='pyarrow', compression='snappy')
                    processed_count += 1
                    logger.info(f"Saved processed data for building {building_id}")
                
                # Clean up
                del building_data
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to process building {building_id}: {traceback.format_exc()}")
                continue
                
        return processed_count

    

def main():
    try:
        # Get project paths
        project_root, data_dir = get_project_paths()
        
        # Create output directory
        output_dir = os.path.join(data_dir, 'processed_minimal')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize preprocessor
        preprocessor = MinimalPreprocessor(data_dir)
        
        # Use first week of data from the actual start date
        start_date = preprocessor.start_date
        end_date = start_date + pd.Timedelta(days=7)
        
        logger.info(f"Processing data from {start_date} to {end_date}")
        
        # Process buildings that have data
        valid_buildings = preprocessor.valid_buildings[:10]  # Start with 10 buildings
        
        logger.info(f"Processing {len(valid_buildings)} buildings")
        
        # Process in chunks of 2
        for i in range(0, len(valid_buildings), 2):
            chunk_buildings = valid_buildings[i:i+2]
            logger.info(f"Processing chunk {i//2 + 1}")
            
            processed = preprocessor.process_chunk(
                chunk_buildings,
                start_date,
                end_date,
                output_dir
            )
            
            gc.collect()
            log_memory_usage()
        
    except Exception as e:
        logger.error(f"Fatal error: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()