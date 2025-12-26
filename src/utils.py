"""
Common utility functions for CIC IoMT 2024 project.
Contains shared functions for data loading, preprocessing, and common operations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def get_project_paths():
    """
    Get common project paths.
    
    Returns:
    --------
    dict: Dictionary containing BASE_DIR, DATA_DIR, OUTPUT_DIR, MODELS_DIR
    """
    BASE_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
    
    paths = {
        'BASE_DIR': BASE_DIR,
        'DATA_DIR': BASE_DIR / 'dataset' / 'value',
        'OUTPUT_DIR': BASE_DIR / 'dataset' / 'processed',
        'MODELS_DIR': BASE_DIR / 'models',
        'RESULTS_DIR': BASE_DIR / 'results'
    }
    
    # Create directories if they don't exist
    paths['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)
    paths['MODELS_DIR'].mkdir(parents=True, exist_ok=True)
    paths['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)
    
    return paths


def load_datasets(data_dir=None):
    """
    Load training and test datasets from parquet files.
    
    Parameters:
    -----------
    data_dir : Path, optional
        Directory containing the dataset files. If None, uses default DATA_DIR.
    
    Returns:
    --------
    tuple: (train_df, test_df) pandas DataFrames
    """
    if data_dir is None:
        paths = get_project_paths()
        data_dir = paths['DATA_DIR']
    
    train_df = pd.read_parquet(data_dir / 'CIC_IoMT_2024_WiFi_MQTT_train.parquet')
    test_df = pd.read_parquet(data_dir / 'CIC_IoMT_2024_WiFi_MQTT_test.parquet')
    
    return train_df, test_df


def remove_exact_duplicates(df, keep='first', verbose=True, return_stats=False):
    """
    Remove exact duplicate rows (same features and labels) from a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to clean
    keep : {'first', 'last', False}, default='first'
        - 'first': Keep first occurrence of duplicates
        - 'last': Keep last occurrence of duplicates
        - False: Remove all duplicates (including first occurrence)
    verbose : bool, default=True
        If True, prints statistics about duplicates removed
    return_stats : bool, default=False
        If True, returns a dictionary with statistics
    
    Returns:
    --------
    pandas.DataFrame or tuple
        Cleaned dataframe (and optionally statistics dictionary)
    """
    # Store original shape
    original_shape = df.shape
    original_count = len(df)
    
    # Count duplicates before removal
    duplicate_count = df.duplicated().sum()
    
    # Remove exact duplicates
    df_clean = df.drop_duplicates(keep=keep)
    
    # Calculate statistics
    removed_count = original_count - len(df_clean)
    removed_percentage = (removed_count / original_count) * 100 if original_count > 0 else 0
    
    stats = {
        'original_rows': original_count,
        'duplicate_rows': duplicate_count,
        'removed_rows': removed_count,
        'remaining_rows': len(df_clean),
        'removed_percentage': removed_percentage,
        'original_shape': original_shape,
        'cleaned_shape': df_clean.shape
    }
    
    # Print statistics if verbose
    if verbose:
        print("=" * 80)
        print("DUPLICATE REMOVAL SUMMARY")
        print("=" * 80)
        print(f"Original rows: {original_count:,}")
        print(f"Duplicate rows found: {duplicate_count:,}")
        print(f"Removed duplicates: {removed_count:,} ({removed_percentage:.2f}%)")
        print(f"Remaining unique rows: {len(df_clean):,}")
        print(f"Original shape: {original_shape}")
        print(f"Cleaned shape: {df_clean.shape}")
        print("=" * 80)
    
    if return_stats:
        return df_clean, stats
    else:
        return df_clean


def comprehensive_outlier_check(df, columns, methods=['iqr', 'zscore']):
    """
    Comprehensive outlier detection using multiple methods.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of column names to check
    methods : list
        Methods to use ['iqr', 'zscore']
    
    Returns:
    --------
    dict: Dictionary with outlier counts and percentages per method
    """
    results = {}
    total_samples = len(df)
    
    for col in columns:
        results[col] = {}
        
        if 'iqr' in methods:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Avoid division by zero
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                iqr_outliers = len(df[(df[col] < lower) | (df[col] > upper)])
                results[col]['iqr'] = {
                    'count': iqr_outliers,
                    'percentage': (iqr_outliers / total_samples) * 100
                }
            else:
                results[col]['iqr'] = {'count': 0, 'percentage': 0.0}
        
        if 'zscore' in methods:
            try:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                zscore_outliers = len(z_scores[z_scores > 3])
                results[col]['zscore'] = {
                    'count': zscore_outliers,
                    'percentage': (zscore_outliers / total_samples) * 100
                }
            except:
                results[col]['zscore'] = {'count': 0, 'percentage': 0.0}
    
    return results

