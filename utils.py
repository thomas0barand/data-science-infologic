"""
Utility functions for loading and cleaning data from the Copilote user traces dataset.
"""

import pandas as pd
import numpy as np
import os
import re
from collections import Counter


def load_data(ds_name: str, data_dir: str = "data"):
    """
    Load a variable-column data file from CSV files in the /data folder.
    
    This function reads user session data where:
    - TRAIN files: First column is user ID, second is browser, then session actions
    - TEST files: First column is browser, then session actions (no user ID)
    
    The data contains user interaction traces with the Copilote application,
    including actions like button clicks, form entries, screen selections, etc.
    Time markers (t5, t10, t15...) indicate 5-second intervals.
    
    Args:
        ds_name (str): Dataset name ('train' or 'test')
        data_dir (str): Directory containing the data files
        
    Returns:
        pd.DataFrame: DataFrame with properly named columns and padded rows
    """
    # Construct file path
    base_dir = os.path.join(os.path.dirname(__file__), data_dir)
    fname = os.path.join(base_dir, f"{ds_name}.csv")
    
    # Read file line by line to handle variable number of columns
    with open(fname, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # Split each line by comma delimiter
    split_lines = [line.split(",") for line in lines]
    
    # Find maximum number of columns for padding
    max_len = max(len(line) for line in split_lines)

    # Pad shorter rows with None values to ensure consistent DataFrame structure
    padded_lines = [line + [None] * (max_len - len(line)) for line in split_lines]

    # Create column names based on dataset type
    if ds_name == "train":
        # Train: user_id, browser, then actions
        columns = ["user_id", "browser"] + [f"action_{i}" for i in range(1, max_len - 1)]
    else:
        # Test: browser, then actions (no user_id)
        columns = ["browser"] + [f"action_{i}" for i in range(1, max_len)]

    # Create and return DataFrame
    df = pd.DataFrame(padded_lines, columns=columns)
    return df


def clean_data(df, is_train=True):
    """
    Clean and preprocess the data.
    
    Args:
        df (pd.DataFrame): Raw dataframe from load_data
        is_train (bool): Whether this is training data (has user_id column)
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # For training data, encode user_id as categorical
    if is_train and "user_id" in df_clean.columns:
        df_clean["user_id_cat"] = pd.Categorical(df_clean["user_id"])
        df_clean["user_id_code"] = df_clean["user_id_cat"].cat.codes
    
    return df_clean


def extract_features(df, is_train=True):
    """
    Extract features from the raw action data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        is_train (bool): Whether this is training data
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    features = pd.DataFrame()
    
    # Get action columns
    action_cols = [col for col in df.columns if col.startswith('action_')]
    
    # Feature 1: Total number of non-null actions
    features['num_actions'] = df[action_cols].notna().sum(axis=1)
    
    # Feature 2: Session duration (count time markers)
    def count_time_markers(row):
        time_markers = [val for val in row if val and str(val).startswith('t')]
        return len(time_markers)
    
    features['session_duration'] = df[action_cols].apply(count_time_markers, axis=1)
    
    # Feature 3: Count specific action types
    def count_action_type(row, action_keyword):
        count = 0
        for val in row:
            if val and action_keyword in str(val):
                count += 1
        return count
    
    # Common actions to count
    features['num_button_exec'] = df[action_cols].apply(lambda row: count_action_type(row, "Exécution d'un bouton"), axis=1)
    features['num_dialog_display'] = df[action_cols].apply(lambda row: count_action_type(row, "Affichage d'une dialogue"), axis=1)
    features['num_dialog_close'] = df[action_cols].apply(lambda row: count_action_type(row, "Fermeture d'une dialogue"), axis=1)
    features['num_field_entry'] = df[action_cols].apply(lambda row: count_action_type(row, "Saisie dans un champ"), axis=1)
    features['num_double_click'] = df[action_cols].apply(lambda row: count_action_type(row, "Double-clic"), axis=1)
    features['num_screen_creation'] = df[action_cols].apply(lambda row: count_action_type(row, "Création d'un écran"), axis=1)
    features['num_toast_display'] = df[action_cols].apply(lambda row: count_action_type(row, "Affichage d'un toast"), axis=1)
    features['num_filter_sort'] = df[action_cols].apply(lambda row: count_action_type(row, "Filtrage / Tri"), axis=1)
    
    # Feature 4: Extract most common screen using regex pattern
    pattern_ecran = re.compile(r"\((.*?)\)")
    
    def get_most_common_screen(row):
        screens = []
        for val in row:
            if val:
                matches = pattern_ecran.findall(str(val))
                screens.extend(matches)
        if screens:
            most_common = Counter(screens).most_common(1)[0][0]
            return most_common
        return "none"
    
    features['most_common_screen'] = df[action_cols].apply(get_most_common_screen, axis=1)
    
    # Feature 5: Extract most common chain category using regex pattern
    pattern_chaine = re.compile(r"\$(.*?)\$")
    
    def get_most_common_chain(row):
        chains = []
        for val in row:
            if val:
                matches = pattern_chaine.findall(str(val))
                chains.extend(matches)
        if chains:
            most_common = Counter(chains).most_common(1)[0][0]
            return most_common
        return "none"
    
    features['most_common_chain'] = df[action_cols].apply(get_most_common_chain, axis=1)
    
    # Feature 6: Browser type (one-hot encoding)
    browser_dummies = pd.get_dummies(df['browser'], prefix='browser')
    features = pd.concat([features, browser_dummies], axis=1)
    
    # Feature 7: Actions per time unit (if session_duration > 0)
    features['actions_per_time'] = features.apply(
        lambda row: row['num_actions'] / row['session_duration'] if row['session_duration'] > 0 else 0,
        axis=1
    )
    
    return features


def prepare_training_data(df):
    """
    Prepare training data by combining features and target.
    
    Args:
        df (pd.DataFrame): Cleaned training dataframe
        
    Returns:
        tuple: (X_features, y_target, user_categories)
    """
    # Extract features
    X = extract_features(df, is_train=True)
    
    # Get target variable
    user_id_cat = pd.Categorical(df["user_id"])
    y = user_id_cat.codes
    
    # Handle categorical features (convert to codes)
    for col in ['most_common_screen', 'most_common_chain']:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes
    
    return X, y, user_id_cat


def prepare_test_data(df, train_features_columns):
    """
    Prepare test data to match training data structure.
    
    Args:
        df (pd.DataFrame): Cleaned test dataframe
        train_features_columns (list): List of column names from training features
        
    Returns:
        pd.DataFrame: Features matching training data structure
    """
    # Extract features
    X = extract_features(df, is_train=False)
    
    # Handle categorical features (convert to codes)
    for col in ['most_common_screen', 'most_common_chain']:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes
    
    # Ensure all columns from training are present
    for col in train_features_columns:
        if col not in X.columns:
            X[col] = 0  # Add missing columns with default value
    
    # Reorder columns to match training data
    X = X[train_features_columns]
    
    return X

