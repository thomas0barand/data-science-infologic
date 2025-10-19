"""
Caching utilities for preprocessed data.

This module provides functions to cache and load preprocessed sequences,
features, and vocabularies to speed up training iterations.
"""

import os
import hashlib
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any


def compute_data_hash(config: Any, data_size: str) -> str:
    """
    Compute a hash for the data configuration to create unique cache keys.
    
    Args:
        config: Hydra configuration object
        data_size: Size of dataset ('full' or 'small')
        
    Returns:
        str: Hash string for cache identification
    """
    # Create a string from relevant config parameters
    cache_key = f"{data_size}_{config.data.max_sequence_length}_{config.seed}"
    
    # Generate hash
    return hashlib.md5(cache_key.encode()).hexdigest()[:12]


def get_cache_dir(base_dir: str = "cache") -> str:
    """
    Get or create the cache directory.
    
    Args:
        base_dir: Base directory for cache
        
    Returns:
        str: Path to cache directory
    """
    cache_dir = os.path.abspath(base_dir)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cache_paths(config: Any, cache_dir: str = "cache") -> Dict[str, str]:
    """
    Generate cache file paths for all preprocessing stages.
    
    Args:
        config: Hydra configuration object
        cache_dir: Cache directory path
        
    Returns:
        dict: Dictionary with cache file paths for each stage
    """
    cache_dir = get_cache_dir(cache_dir)
    cache_hash = compute_data_hash(config, config.data.size)
    
    return {
        # Stage 1: Tokenized sequences and vocabulary
        'sequences': os.path.join(cache_dir, f'sequences_{cache_hash}.joblib'),
        'vocabulary': os.path.join(cache_dir, f'vocabulary_{cache_hash}.joblib'),
        
        # Stage 2: Prepared RNN sequences
        'X_sequences': os.path.join(cache_dir, f'X_sequences_{cache_hash}.npy'),
        'X_browser': os.path.join(cache_dir, f'X_browser_{cache_hash}.npy'),
        'y': os.path.join(cache_dir, f'y_{cache_hash}.npy'),
        'user_categories': os.path.join(cache_dir, f'user_categories_{cache_hash}.joblib'),
        
        # Stage 3: Statistical features
        'X_stat': os.path.join(cache_dir, f'X_stat_{cache_hash}.npy'),
        
        # Stage 4: Cleaned dataframe (optional - can be large)
        'df_train_clean': os.path.join(cache_dir, f'df_train_clean_{cache_hash}.pkl'),
        
        # Metadata
        'metadata': os.path.join(cache_dir, f'metadata_{cache_hash}.joblib')
    }


def save_stage1_cache(sequences: list, vocabulary: dict, cache_paths: Dict[str, str]) -> None:
    """
    Save Stage 1: Tokenized sequences and vocabulary.
    
    Args:
        sequences: List of tokenized sequences
        vocabulary: Action vocabulary dictionary
        cache_paths: Dictionary of cache file paths
    """
    print(f"💾 Saving Stage 1 cache (sequences & vocabulary)...")
    joblib.dump(sequences, cache_paths['sequences'])
    joblib.dump(vocabulary, cache_paths['vocabulary'])
    print(f"   ✓ Saved to {os.path.dirname(cache_paths['sequences'])}")


def load_stage1_cache(cache_paths: Dict[str, str]) -> Optional[Tuple[list, dict]]:
    """
    Load Stage 1: Tokenized sequences and vocabulary.
    
    Args:
        cache_paths: Dictionary of cache file paths
        
    Returns:
        tuple: (sequences, vocabulary) or None if cache doesn't exist
    """
    if os.path.exists(cache_paths['sequences']) and os.path.exists(cache_paths['vocabulary']):
        print(f"📦 Loading Stage 1 cache (sequences & vocabulary)...")
        sequences = joblib.load(cache_paths['sequences'])
        vocabulary = joblib.load(cache_paths['vocabulary'])
        print(f"   ✓ Loaded from cache")
        return sequences, vocabulary
    return None


def save_stage2_cache(X_sequences: np.ndarray, X_browser: np.ndarray, 
                      y: np.ndarray, user_categories: pd.Categorical,
                      cache_paths: Dict[str, str]) -> None:
    """
    Save Stage 2: Prepared RNN sequences.
    
    Args:
        X_sequences: Padded sequence array
        X_browser: Browser features
        y: Target labels
        user_categories: User categorical mapping
        cache_paths: Dictionary of cache file paths
    """
    print(f"💾 Saving Stage 2 cache (RNN sequences)...")
    np.save(cache_paths['X_sequences'], X_sequences)
    np.save(cache_paths['X_browser'], X_browser)
    np.save(cache_paths['y'], y)
    joblib.dump(user_categories, cache_paths['user_categories'])
    print(f"   ✓ Saved to {os.path.dirname(cache_paths['X_sequences'])}")


def load_stage2_cache(cache_paths: Dict[str, str]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Categorical]]:
    """
    Load Stage 2: Prepared RNN sequences.
    
    Args:
        cache_paths: Dictionary of cache file paths
        
    Returns:
        tuple: (X_sequences, X_browser, y, user_categories) or None if cache doesn't exist
    """
    required_files = ['X_sequences', 'X_browser', 'y', 'user_categories']
    if all(os.path.exists(cache_paths[f]) for f in required_files):
        print(f"📦 Loading Stage 2 cache (RNN sequences)...")
        X_sequences = np.load(cache_paths['X_sequences'])
        X_browser = np.load(cache_paths['X_browser'])
        y = np.load(cache_paths['y'])
        user_categories = joblib.load(cache_paths['user_categories'])
        print(f"   ✓ Loaded from cache")
        return X_sequences, X_browser, y, user_categories
    return None


def save_stage3_cache(X_stat: np.ndarray, cache_paths: Dict[str, str]) -> None:
    """
    Save Stage 3: Statistical features.
    
    Args:
        X_stat: Statistical features array
        cache_paths: Dictionary of cache file paths
    """
    print(f"💾 Saving Stage 3 cache (statistical features)...")
    np.save(cache_paths['X_stat'], X_stat)
    print(f"   ✓ Saved to {os.path.dirname(cache_paths['X_stat'])}")


def load_stage3_cache(cache_paths: Dict[str, str]) -> Optional[np.ndarray]:
    """
    Load Stage 3: Statistical features.
    
    Args:
        cache_paths: Dictionary of cache file paths
        
    Returns:
        np.ndarray: Statistical features or None if cache doesn't exist
    """
    if os.path.exists(cache_paths['X_stat']):
        print(f"📦 Loading Stage 3 cache (statistical features)...")
        X_stat = np.load(cache_paths['X_stat'])
        print(f"   ✓ Loaded from cache")
        return X_stat
    return None


def save_stage4_cache(df_train_clean: pd.DataFrame, cache_paths: Dict[str, str]) -> None:
    """
    Save Stage 4: Cleaned dataframe (optional - can be large).
    
    Args:
        df_train_clean: Cleaned training dataframe
        cache_paths: Dictionary of cache file paths
    """
    print(f"💾 Saving Stage 4 cache (cleaned dataframe)...")
    df_train_clean.to_pickle(cache_paths['df_train_clean'])
    print(f"   ✓ Saved to {os.path.dirname(cache_paths['df_train_clean'])}")


def load_stage4_cache(cache_paths: Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    Load Stage 4: Cleaned dataframe.
    
    Args:
        cache_paths: Dictionary of cache file paths
        
    Returns:
        pd.DataFrame: Cleaned dataframe or None if cache doesn't exist
    """
    if os.path.exists(cache_paths['df_train_clean']):
        print(f"📦 Loading Stage 4 cache (cleaned dataframe)...")
        df_train_clean = pd.read_pickle(cache_paths['df_train_clean'])
        print(f"   ✓ Loaded from cache")
        return df_train_clean
    return None


def save_metadata(metadata: dict, cache_paths: Dict[str, str]) -> None:
    """
    Save metadata about cached data.
    
    Args:
        metadata: Dictionary with metadata
        cache_paths: Dictionary of cache file paths
    """
    joblib.dump(metadata, cache_paths['metadata'])


def load_metadata(cache_paths: Dict[str, str]) -> Optional[dict]:
    """
    Load metadata about cached data.
    
    Args:
        cache_paths: Dictionary of cache file paths
        
    Returns:
        dict: Metadata or None if doesn't exist
    """
    if os.path.exists(cache_paths['metadata']):
        return joblib.load(cache_paths['metadata'])
    return None


def check_cache_complete(cache_paths: Dict[str, str], stages: list = None) -> bool:
    """
    Check if all required cache files exist.
    
    Args:
        cache_paths: Dictionary of cache file paths
        stages: List of stages to check (e.g., [1, 2, 3]). If None, checks all.
        
    Returns:
        bool: True if all required files exist
    """
    if stages is None:
        stages = [1, 2, 3]
    
    required_files = []
    
    if 1 in stages:
        required_files.extend(['sequences', 'vocabulary'])
    if 2 in stages:
        required_files.extend(['X_sequences', 'X_browser', 'y', 'user_categories'])
    if 3 in stages:
        required_files.append('X_stat')
    if 4 in stages:
        required_files.append('df_train_clean')
    
    return all(os.path.exists(cache_paths[f]) for f in required_files)


def clear_cache(cache_dir: str = "cache", pattern: str = None) -> None:
    """
    Clear cache files.
    
    Args:
        cache_dir: Cache directory path
        pattern: Optional pattern to match files (e.g., specific hash)
    """
    cache_dir = get_cache_dir(cache_dir)
    
    if not os.path.exists(cache_dir):
        print(f"No cache directory found at {cache_dir}")
        return
    
    files_removed = 0
    for filename in os.listdir(cache_dir):
        if pattern is None or pattern in filename:
            filepath = os.path.join(cache_dir, filename)
            try:
                os.remove(filepath)
                files_removed += 1
            except Exception as e:
                print(f"⚠️  Could not remove {filepath}: {e}")
    
    print(f"✓ Removed {files_removed} cache files from {cache_dir}")


def print_cache_info(cache_paths: Dict[str, str]) -> None:
    """
    Print information about cache status.
    
    Args:
        cache_paths: Dictionary of cache file paths
    """
    print("\n" + "=" * 80)
    print("📂 Cache Status")
    print("=" * 80)
    
    stages = {
        'Stage 1 (Sequences & Vocabulary)': ['sequences', 'vocabulary'],
        'Stage 2 (RNN Sequences)': ['X_sequences', 'X_browser', 'y', 'user_categories'],
        'Stage 3 (Statistical Features)': ['X_stat'],
        'Stage 4 (Cleaned Dataframe)': ['df_train_clean']
    }
    
    total_size = 0
    
    for stage_name, files in stages.items():
        stage_exists = all(os.path.exists(cache_paths[f]) for f in files)
        stage_size = 0
        
        if stage_exists:
            for f in files:
                if os.path.exists(cache_paths[f]):
                    stage_size += os.path.getsize(cache_paths[f])
            
            status = f"✓ Cached ({stage_size / (1024**2):.2f} MB)"
            total_size += stage_size
        else:
            status = "✗ Not cached"
        
        print(f"{stage_name:40s} {status}")
    
    print(f"\nTotal cache size: {total_size / (1024**2):.2f} MB")
    print(f"Cache directory: {os.path.dirname(list(cache_paths.values())[0])}")
    print("=" * 80 + "\n")

