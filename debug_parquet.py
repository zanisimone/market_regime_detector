"""
Debug script to examine parquet file structure.
"""

import pandas as pd
from pathlib import Path

def examine_parquet_file(file_path: Path):
    """Examine a parquet file structure."""
    print(f"🔍 Examining: {file_path}")
    print("=" * 50)
    
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Successfully loaded parquet file")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:")
        print(df.dtypes)
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nSample values:")
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
        
        # Check for date column
        if 'date' in df.columns:
            print(f"\nDate column info:")
            print(f"  Type: {df['date'].dtype}")
            print(f"  Sample: {df['date'].iloc[0] if len(df) > 0 else 'N/A'}")
            print(f"  Min: {df['date'].min() if len(df) > 0 else 'N/A'}")
            print(f"  Max: {df['date'].max() if len(df) > 0 else 'N/A'}")
        
        # Check for regime column
        if 'regime' in df.columns:
            print(f"\nRegime column info:")
            print(f"  Type: {df['regime'].dtype}")
            print(f"  Unique values: {sorted(df['regime'].unique()) if len(df) > 0 else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Error loading parquet: {e}")

def main():
    """Examine all parquet files."""
    reports_dir = Path("reports")
    
    for model_dir in reports_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        print(f"\n📁 Model: {model_dir.name}")
        print("=" * 60)
        
        # Check train_labels.parquet
        train_file = model_dir / "train_labels.parquet"
        if train_file.exists():
            examine_parquet_file(train_file)
        
        # Check test_labels.parquet
        test_file = model_dir / "test_labels.parquet"
        if test_file.exists():
            examine_parquet_file(test_file)
        
        # Check rolling_labels.parquet
        rolling_file = model_dir / "rolling_labels.parquet"
        if rolling_file.exists():
            examine_parquet_file(rolling_file)
        
        # Check kmeans_labels.csv
        csv_file = model_dir / f"{model_dir.name}_labels.csv"
        if csv_file.exists():
            print(f"\n📄 CSV file: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"First 5 rows:")
                print(df.head())
            except Exception as e:
                print(f"❌ Error loading CSV: {e}")

if __name__ == "__main__":
    main()
