import pandas as pd
import logging
from pathlib import Path

class ElectricityDemandProcessor:
    def __init__(self, raw_data_dir: str, processed_data_path: str, chunk_size: int = 500_000):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_path = Path(processed_data_path)
        self.chunk_size = chunk_size

        # Ensure the output directory exists
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    def load_metadata(self) -> pd.DataFrame:
        """Load and clean metadata."""
        logging.info("Loading metadata.parquet...")
        metadata_path = self.raw_data_dir / "metadata.parquet"
        metadata = pd.read_parquet(metadata_path)
        metadata["location_id"] = metadata["location_id"].fillna("UNKNOWN")
        logging.info("Loaded and cleaned metadata.parquet")
        return metadata

    def load_weather(self) -> pd.DataFrame:
        """Load weather data."""
        logging.info("Loading weather.parquet...")
        weather_path = self.raw_data_dir / "weather.parquet"
        weather = pd.read_parquet(weather_path)
        logging.info("Loaded weather.parquet")
        return weather

    def detect_and_handle_outliers(self, df: pd.DataFrame, column: str, threshold_multiplier: float = 1.5):
        """Detect and cap outliers in the specified column."""
        logging.info("Detecting and handling outliers...")
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + threshold_multiplier * iqr
        df.loc[:, column] = df[column].clip(upper=upper_limit)
        logging.info(f"Outliers capped for column '{column}'.")
        return df

    def process_demand_chunk(self, demand_chunk: pd.DataFrame, metadata: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of demand data."""
        logging.info("Processing a demand chunk...")

        # Handle nulls in demand
        demand_chunk.loc[:, "y"] = demand_chunk["y"].fillna(0)

        # Detect and handle outliers in demand
        demand_chunk = self.detect_and_handle_outliers(demand_chunk, "y")

        # Join demand with metadata
        demand_chunk = demand_chunk.merge(metadata, on="unique_id", how="left")

        # Join with weather data
        demand_chunk = demand_chunk.merge(weather, on=["location_id", "timestamp"], how="left")

        return demand_chunk

    def process(self):
        logging.info("Starting data processing pipeline...")

        # Load metadata and weather
        metadata = self.load_metadata()
        weather = self.load_weather()

        # Read demand dataset manually in chunks
        demand_path = self.raw_data_dir / "demand.parquet"
        demand_data = pd.read_parquet(demand_path)

        total_rows = len(demand_data)
        logging.info(f"Total rows in demand data: {total_rows}")

        all_chunks = []
        for start_row in range(0, total_rows, self.chunk_size):
            end_row = min(start_row + self.chunk_size, total_rows)
            demand_chunk = demand_data.iloc[start_row:end_row]
            logging.info(f"Processing demand chunk {start_row} to {end_row}...")

            processed_chunk = self.process_demand_chunk(demand_chunk, metadata, weather)

            # Append processed chunk
            all_chunks.append(processed_chunk)

            # Write to disk every 10 chunks
            if len(all_chunks) >= 10:
                combined = pd.concat(all_chunks)
                if not self.processed_data_path.exists():
                    combined.to_parquet(self.processed_data_path, engine="pyarrow", index=False)
                else:
                    combined.to_parquet(self.processed_data_path, engine="pyarrow", index=False)
                all_chunks.clear()

        # Write any remaining chunks
        if all_chunks:
            combined = pd.concat(all_chunks)
            if not self.processed_data_path.exists():
                combined.to_parquet(self.processed_data_path, engine="pyarrow", index=False)
            else:
                combined.to_parquet(self.processed_data_path, engine="pyarrow", index=False)

        logging.info(f"Processing complete. Processed data saved at {self.processed_data_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    raw_data_dir = "C:/Users/prite/Desktop/mlops-timeseries/data/raw"
    processed_data_path = "C:/Users/prite/Desktop/mlops-timeseries/data/processed/processed_data.parquet"

    processor = ElectricityDemandProcessor(raw_data_dir, processed_data_path, chunk_size=500_000)
    try:
        processor.process()
    except Exception as e:
        logging.error(f"Processing failed: {e}")
