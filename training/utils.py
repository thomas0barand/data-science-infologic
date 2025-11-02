"""
Utility functions for loading and cleaning data from the Copilote user traces dataset.
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from collections import Counter

import json
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def load_data(ds_name: str, data_dir: str = "data", size: str = "full"):
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
        size (str): Size of the dataset ('full' or 'small')
            - 'full': Load all data
            - 'small': For train data, load all rows for 20% of users
                      For test data, load first 20% of rows
    Returns:
        pd.DataFrame: DataFrame with properly named columns and padded rows
    """
    # Construct file path
    fname = os.path.join(data_dir, f"{ds_name}.csv")
    
    # Read file line by line to handle variable number of columns
    with open(fname, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # Split each line by comma delimiter
    split_lines = [line.split(",") for line in lines]
    
    # For training data with 'small' size, filter to 20% of users
    if size == "small" and ds_name == "train":
        # Extract user IDs (first column)
        user_ids = [line[0] for line in split_lines]
        
        # Get unique users and sort for deterministic selection
        unique_users = sorted(set(user_ids))
        
        # Select first 20% of users
        num_users_to_select = max(1, int(len(unique_users) * 0.2))
        selected_users = set(unique_users[:num_users_to_select])
        
        # Filter to keep only rows for selected users
        split_lines = [line for line in split_lines if line[0] in selected_users]
        
        print(f"   ℹ️  Small dataset: Selected {num_users_to_select} out of {len(unique_users)} users (20%)")
        print(f"   ℹ️  Total rows: {len(split_lines)}")
    elif size == "small" and ds_name == "test":
        # For test data, take first 20% of rows
        num_rows = max(1, int(len(split_lines) * 0.2))
        split_lines = split_lines[:num_rows]
        print(f"   ℹ️  Small dataset: Selected {num_rows} out of {len(lines)} rows (20%)")
    
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
    
    These features are designed to capture USER BEHAVIOR PATTERNS:
    - Activity level (how much they do)
    - Work style (fast vs methodical, exploration vs focused)
    - Preferred workflows and modules
    - Interaction patterns (mouse vs keyboard, errors, etc.)
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        is_train (bool): Whether this is training data
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    features = pd.DataFrame()
    
    # Get action columns
    action_cols = [col for col in df.columns if col.startswith('action_')]
    
    # ============================================================================
    # BASIC ACTIVITY FEATURES
    # ============================================================================
    
    # Feature 1: Total number of non-null actions (activity level)
    features['num_actions'] = df[action_cols].notna().sum(axis=1)
    
    # Feature 2: Session duration (count time markers)
    def count_time_markers(row):
        time_markers = [val for val in row if val and str(val).startswith('t')]
        return len(time_markers)
    
    features['session_duration'] = df[action_cols].apply(count_time_markers, axis=1)
    
    # Feature 3: Actions per time unit (work pace)
    features['actions_per_time'] = features.apply(
        lambda row: row['num_actions'] / row['session_duration'] if row['session_duration'] > 0 else 0,
        axis=1
    )
    
    # ============================================================================
    # ACTION TYPE COUNTS (behavioral patterns)
    # ============================================================================
    
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
    
    # NEW: More specific action types
    features['num_screen_selection'] = df[action_cols].apply(lambda row: count_action_type(row, "Sélection d'un écran"), axis=1)
    features['num_tab_selection'] = df[action_cols].apply(lambda row: count_action_type(row, "Sélection d'un onglet"), axis=1)
    features['num_errors'] = df[action_cols].apply(lambda row: count_action_type(row, "Affichage d'une erreur"), axis=1)
    features['num_generic_actions'] = df[action_cols].apply(lambda row: count_action_type(row, "Lancement d'une action générique"), axis=1)
    features['num_stats'] = df[action_cols].apply(lambda row: count_action_type(row, "Lancement d'une stat"), axis=1)
    features['num_shortcuts'] = df[action_cols].apply(lambda row: count_action_type(row, "Raccourci"), axis=1)
    features['num_table_actions'] = df[action_cols].apply(lambda row: count_action_type(row, "Action de table"), axis=1)
    features['num_chaining'] = df[action_cols].apply(lambda row: count_action_type(row, "Chainage"), axis=1)
    
    # ============================================================================
    # RATIO FEATURES (work style indicators)
    # ============================================================================
    
    # Ratio of dialogs closed vs opened (organized vs messy)
    features['dialog_close_ratio'] = features.apply(
        lambda row: row['num_dialog_close'] / row['num_dialog_display'] if row['num_dialog_display'] > 0 else 0,
        axis=1
    )
    
    # Ratio of keyboard entry vs mouse clicks (power user indicator)
    features['keyboard_vs_mouse'] = features.apply(
        lambda row: row['num_field_entry'] / (row['num_button_exec'] + 1),  # +1 to avoid division by zero
        axis=1
    )
    
    # Error rate (user expertise level)
    features['error_rate'] = features.apply(
        lambda row: row['num_errors'] / row['num_actions'] if row['num_actions'] > 0 else 0,
        axis=1
    )
    
    # ============================================================================
    # DIVERSITY FEATURES (exploration vs focused work)
    # ============================================================================
    
    def count_unique_non_time_actions(row):
        """Count how many different action types (not time markers) the user performs"""
        actions = [str(val) for val in row if val and not str(val).startswith('t')]
        unique_actions = set([action.split('(')[0].split('<')[0].split('$')[0] for action in actions])
        return len(unique_actions)
    
    features['action_diversity'] = df[action_cols].apply(count_unique_non_time_actions, axis=1)
    
    # ============================================================================
    # WORKFLOW & MODULE FEATURES (what they work on)
    # ============================================================================
    
    # Extract most common screen using regex pattern
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
    
    # Count unique screens used (module diversity)
    def count_unique_screens(row):
        screens = []
        for val in row:
            if val:
                matches = pattern_ecran.findall(str(val))
                screens.extend(matches)
        return len(set(screens))
    
    features['num_unique_screens'] = df[action_cols].apply(count_unique_screens, axis=1)
    
    # Extract most common chain category using regex pattern
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
    
    # Count unique chains (workflow diversity)
    def count_unique_chains(row):
        chains = []
        for val in row:
            if val:
                matches = pattern_chaine.findall(str(val))
                chains.extend(matches)
        return len(set(chains))
    
    features['num_unique_chains'] = df[action_cols].apply(count_unique_chains, axis=1)
    
    # ============================================================================
    # CONFIGURATION FEATURES
    # ============================================================================
    
    # Browser type (one-hot encoding)
    browser_dummies = pd.get_dummies(df['browser'], prefix='browser')
    features = pd.concat([features, browser_dummies], axis=1)
    
    # ============================================================================
    # SEQUENCE FEATURES (action patterns over time)
    # ============================================================================
    
    # First action in session (startup behavior)
    def get_first_action(row):
        for val in row:
            if val and not str(val).startswith('t'):
                return str(val).split('(')[0].split('<')[0].split('$')[0]
        return "none"
    
    features['first_action'] = df[action_cols].apply(get_first_action, axis=1)
    
    # Average time between actions
    features['avg_time_between_actions'] = features.apply(
        lambda row: row['session_duration'] / row['num_actions'] if row['num_actions'] > 0 else 0,
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
    categorical_cols = ['most_common_screen', 'most_common_chain', 'first_action']
    for col in categorical_cols:
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
    categorical_cols = ['most_common_screen', 'most_common_chain', 'first_action']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes
    
    # Ensure all columns from training are present
    for col in train_features_columns:
        if col not in X.columns:
            X[col] = 0  # Add missing columns with default value
    
    # Reorder columns to match training data
    X = X[train_features_columns]
    
    return X


# ============================================================================
# RNN SEQUENCE PREPROCESSING FUNCTIONS
# ============================================================================

def parse_action_string(action_str):
    """
    Parse action string to extract action type and its parameters.
    
    Returns a tuple containing (action, paren_content, angle_content) which
    creates a unique signature for each action string pattern.
    
    Examples:
        "Lancement d'une stat(infologic.core.gui.controllers.nested.homeview.InputFormHomeView)"
        -> ("Lancement d'une stat", "infologic.core.gui.controllers.nested.homeview.InputFormHomeView", None)
        
        "Exécution d'un bouton<DEF_03/24>"
        -> ("Exécution d'un bouton", None, "DEF_03/24")
        
        "Saisie dans un champ$JCP$"
        -> ("Saisie dans un champ", None, None)
    
    Args:
        action_str (str): Raw action string
        
    Returns:
        tuple: (action, paren_content, angle_content)
            - action (str): Base action type without parameters
            - paren_content (str or None): Content inside parentheses ()
            - angle_content (str or None): Content inside angle brackets <>
    """
    if not action_str or pd.isna(action_str):
        return (None, None, None)
    
    action_str = str(action_str)
    
    # Extract content in parentheses
    paren_match = re.search(r'\(([^)]*)\)', action_str)
    paren_content = paren_match.group(1) if paren_match else None
    
    # Extract content in angle brackets
    angle_match = re.search(r'<([^>]*)>', action_str)
    angle_content = angle_match.group(1) if angle_match else None
    
    # Extract base action by removing parentheses, angle brackets, and $ content
    action = re.sub(r'\([^)]*\)', '', action_str)
    action = re.sub(r'<[^>]*>', '', action)
    action = re.sub(r'\$[^$]*\$', '', action)
    action = action.strip()
    
    return (action if action else None, paren_content, angle_content)


def tokenize_actions(df, is_train=True, vocabulary=None):
    """
    Tokenize action sequences from dataframe.
    
    Extracts actions, removes time markers, parses action strings,
    and builds/uses vocabulary for integer encoding.
    
    Each action is represented as a tuple (action, paren_content, angle_content)
    which creates a unique signature for each distinct action pattern.
    
    Args:
        df (pd.DataFrame): Dataframe with action columns
        is_train (bool): Whether this is training data
        vocabulary (dict): Existing vocabulary (for test data)
        
    Returns:
        tuple: (sequences, vocabulary)
            - sequences: List of lists of action tuples (action, paren_content, angle_content)
            - vocabulary: Dict mapping action tuple -> integer ID
    """
    # Get action columns
    action_cols = [col for col in df.columns if col.startswith('action_')]
    
    # Convert action columns to numpy array for faster processing
    actions_array = df[action_cols].values
    
    sequences = []
    all_actions = set() if is_train else None
    
    # Process each row using numpy array (much faster than iterrows)
    for row in actions_array:
        sequence = []
        for action in row:
            # Skip None/NaN values
            if pd.isna(action) or action is None:
                continue
            
            action_str = str(action)
            
            # Skip time markers (t5, t10, t15, etc.)
            if action_str.startswith('t') and len(action_str) > 1 and action_str[1:].replace('.', '').isdigit():
                continue
            
            # Parse action to extract base type
            parsed_result = parse_action_string(action_str)
            parsed_action = parsed_result if parsed_result else None
            
            if parsed_action:
                sequence.append(parsed_action)
                if is_train:
                    all_actions.add(parsed_action)
        
        sequences.append(sequence)
    
    # Build or use vocabulary
    if is_train:
        # Reserve 0 for padding, 1 for unknown
        vocabulary = {'<PAD>': 0, '<UNK>': 1}
        # Sort tuples with custom key to handle None values
        # Sort by: (action or '', paren_content or '', angle_content or '')
        sorted_actions = sorted(all_actions, key=lambda x: (x[0] or '', x[1] or '', x[2] or ''))
        for action in sorted_actions:
            vocabulary[action] = len(vocabulary)
    elif vocabulary is None:
        raise ValueError("Vocabulary must be provided for test data")
    
    return sequences, vocabulary


def encode_browser(df):
    """
    One-hot encode browser information.
    
    Args:
        df (pd.DataFrame): Dataframe with 'browser' column
        
    Returns:
        np.ndarray: One-hot encoded browser features (N x num_browsers)
    """
    # Get unique browsers and create encoding
    browser_dummies = pd.get_dummies(df['browser'], prefix='browser')
    
    # Ensure all expected browsers are present
    expected_browsers = ['browser_Firefox', 'browser_Google Chrome', 
                        'browser_Microsoft Edge', 'browser_Opera']
    
    for browser in expected_browsers:
        if browser not in browser_dummies.columns:
            browser_dummies[browser] = 0
    
    # Reorder columns consistently
    browser_dummies = browser_dummies[expected_browsers]
    
    # Convert to float32 to ensure proper dtype for PyTorch
    return browser_dummies.values.astype(np.float32)


def prepare_rnn_sequences(df, vocabulary, max_length=1000, is_train=True):
    """
    Prepare sequences for RNN training/prediction.
    
    Tokenizes actions, truncates to max_length, pads shorter sequences,
    and encodes browser information.
    
    Each action is represented as a unique tuple signature (action, paren_content, angle_content)
    for more granular vocabulary.
    
    Args:
        df (pd.DataFrame): Dataframe with actions and browser
        vocabulary (dict): Action tuple to ID mapping
        max_length (int): Maximum sequence length (default: 1000)
        is_train (bool): Whether this is training data
        
    Returns:
        tuple: (sequences, browser_features, targets, user_categories)
            - sequences: np.ndarray of shape (N, max_length) with action IDs
            - browser_features: np.ndarray of shape (N, num_browsers)
            - targets: np.ndarray of shape (N,) with user IDs (if train)
            - user_categories: Categorical mapping of users (if train)
    """
    # Tokenize actions
    action_sequences, _ = tokenize_actions(df, is_train=False, vocabulary=vocabulary)
    
    # Pre-allocate numpy array for better performance
    num_samples = len(action_sequences)
    sequences = np.zeros((num_samples, max_length), dtype=np.int32)
    pad_id = vocabulary['<PAD>']
    unk_id = vocabulary['<UNK>']
    
    # Convert to integer IDs and pad/truncate
    for i, sequence in enumerate(action_sequences):
        # Convert actions to IDs using list comprehension (faster)
        id_sequence = [vocabulary.get(action, unk_id) for action in sequence]
        
        # Truncate if too long (keep most recent actions)
        if len(id_sequence) > max_length:
            id_sequence = id_sequence[-max_length:]
        
        # Copy to pre-allocated array (already padded with zeros)
        sequences[i, :len(id_sequence)] = id_sequence
    
    # Encode browser information
    browser_features = encode_browser(df)
    
    # Get targets if training data
    if is_train and 'user_id' in df.columns:
        user_id_cat = pd.Categorical(df['user_id'])
        targets = np.array(user_id_cat.codes, dtype=np.int64)
        return sequences, browser_features, targets, user_id_cat
    else:
        return sequences, browser_features, None, None


def extract_statistical_features(df, action_sequences=None):
    """
    Extract statistical behavioral features for deep learning models.
    
    These features capture user behavior patterns that complement
    sequence-based learning. Designed to be used alongside RNN/LSTM models.
    
    Args:
        df (pd.DataFrame): Dataframe with actions and browser
        action_sequences (list, optional): Pre-computed action sequences to avoid recomputation
        
    Returns:
        np.ndarray: Statistical features of shape (N, num_features)
    """
    # Get action columns
    action_cols = [col for col in df.columns if col.startswith('action_')]
    
    features_list = []
    
    # ============================================================================
    # BASIC ACTIVITY METRICS
    # ============================================================================
    
    # 1. Number of actions
    num_actions = df[action_cols].notna().sum(axis=1).values
    features_list.append(num_actions.reshape(-1, 1))
    
    # 2. Session duration (time markers)
    def count_time_markers(row):
        return sum(1 for val in row if val and str(val).startswith('t'))
    
    session_duration = df[action_cols].apply(count_time_markers, axis=1).values
    features_list.append(session_duration.reshape(-1, 1))
    
    # 3. Actions per time unit (work pace)
    actions_per_time = np.divide(num_actions, session_duration, 
                                  out=np.zeros_like(num_actions, dtype=float), 
                                  where=session_duration != 0)
    features_list.append(actions_per_time.reshape(-1, 1))
    
    # ============================================================================
    # ACTION TYPE COUNTS
    # ============================================================================
    
    def count_action_keyword(rows, keyword):
        counts = []
        for row in rows:
            count = sum(1 for val in row if val and keyword in str(val))
            counts.append(count)
        return np.array(counts)
    
    actions_array = df[action_cols].values
    
    # Count various action types
    action_keywords = [
        "Exécution d'un bouton",
        "Affichage d'une dialogue",
        "Fermeture d'une dialogue",
        "Saisie dans un champ",
        "Double-clic",
        "Sélection d'un écran",
        "Sélection d'un onglet",
        "Affichage d'une erreur",
        "Lancement d'une action générique",
        "Raccourci",
        "Filtrage / Tri"
    ]
    
    for keyword in action_keywords:
        counts = count_action_keyword(actions_array, keyword)
        features_list.append(counts.reshape(-1, 1))
    
    # ============================================================================
    # BEHAVIORAL RATIOS
    # ============================================================================
    
    # Dialog close ratio (organized vs messy)
    dialog_open = count_action_keyword(actions_array, "Affichage d'une dialogue")
    dialog_close = count_action_keyword(actions_array, "Fermeture d'une dialogue")
    dialog_ratio = np.divide(dialog_close, dialog_open,
                            out=np.zeros_like(dialog_close, dtype=float),
                            where=dialog_open != 0)
    features_list.append(dialog_ratio.reshape(-1, 1))
    
    # Keyboard vs mouse ratio (power user indicator)
    field_entry = count_action_keyword(actions_array, "Saisie dans un champ")
    button_exec = count_action_keyword(actions_array, "Exécution d'un bouton")
    kb_mouse_ratio = field_entry / (button_exec + 1)
    features_list.append(kb_mouse_ratio.reshape(-1, 1))
    
    # Error rate
    errors = count_action_keyword(actions_array, "Affichage d'une erreur")
    error_rate = np.divide(errors, num_actions,
                          out=np.zeros_like(errors, dtype=float),
                          where=num_actions != 0)
    features_list.append(error_rate.reshape(-1, 1))
    
    # ============================================================================
    # DIVERSITY METRICS
    # ============================================================================
    
    # Action diversity (unique action types)
    def compute_action_diversity(row):
        actions = [str(val).split('(')[0].split('<')[0].split('$')[0] 
                  for val in row if val and not str(val).startswith('t')]
        return len(set(actions))
    
    action_diversity = df[action_cols].apply(compute_action_diversity, axis=1).values
    features_list.append(action_diversity.reshape(-1, 1))
    
    # Screen diversity (unique screens used)
    pattern_screen = re.compile(r"\((.*?)\)")
    
    def count_unique_screens(row):
        screens = []
        for val in row:
            if val:
                screens.extend(pattern_screen.findall(str(val)))
        return len(set(screens))
    
    screen_diversity = df[action_cols].apply(count_unique_screens, axis=1).values
    features_list.append(screen_diversity.reshape(-1, 1))
    
    # ============================================================================
    # TIMING PATTERNS
    # ============================================================================
    
    # Average time between actions
    avg_time_between = np.divide(session_duration, num_actions,
                                 out=np.zeros_like(session_duration, dtype=float),
                                 where=num_actions != 0)
    features_list.append(avg_time_between.reshape(-1, 1))
    
    # ============================================================================
    # SEQUENCE PATTERNS (if action_sequences provided)
    # ============================================================================
    
    if action_sequences is not None:
        # Sequence length (actual non-padded length)
        seq_lengths = np.array([len(seq) for seq in action_sequences])
        features_list.append(seq_lengths.reshape(-1, 1))
        
        # Action bigram diversity (unique consecutive action pairs)
        def count_bigrams(seq):
            if len(seq) < 2:
                return 0
            bigrams = set(zip(seq[:-1], seq[1:]))
            return len(bigrams)
        
        bigram_diversity = np.array([count_bigrams(seq) for seq in action_sequences])
        features_list.append(bigram_diversity.reshape(-1, 1))
    
    # ============================================================================
    # COMBINE ALL FEATURES
    # ============================================================================
    
    # Stack all features horizontally
    statistical_features = np.hstack(features_list).astype(np.float32)
    
    # HOTFIX: Handle any NaN or inf values
    statistical_features = np.nan_to_num(statistical_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # HOTFIX: Additional check - replace any remaining extreme values
    statistical_features = np.clip(statistical_features, -1e6, 1e6)
    
    # Debug: Print if any issues found
    if np.isnan(statistical_features).any() or np.isinf(statistical_features).any():
        print("⚠️  Warning: NaN/Inf found in statistical features after cleaning!")
    
    return statistical_features

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def _ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def plot_class_distribution(df: pd.DataFrame, out_dir: str = "results/figures",
                            filename: str = "class_balance.png"):
    """
    Plot histogram of sessions per user (class balance).
    Requires 'user_id' column (training set).
    """
    _ensure_dir(out_dir)
    if "user_id" not in df.columns:
        print("⚠️ 'user_id' column not found. Skipping class distribution plot.")
        return

    sessions_per_user = df["user_id"].value_counts()
    plt.figure(figsize=(7, 4))
    plt.hist(sessions_per_user.values, bins=30)
    plt.title("Distribution of Sessions per User")
    plt.xlabel("Sessions per user")
    plt.ylabel("Number of users")
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"   ✓ Saved: {save_path}")

def plot_browser_usage(df: pd.DataFrame, out_dir: str = "results/figures",
                       filename: str = "browser_usage.png"):
    """
    Plot bar chart of browser usage counts.
    """
    _ensure_dir(out_dir)
    if "browser" not in df.columns:
        print("⚠️ 'browser' column not found. Skipping browser usage plot.")
        return

    counts = df["browser"].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Browser Usage")
    plt.xlabel("Browser")
    plt.ylabel("Sessions")
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"   ✓ Saved: {save_path}")

def plot_session_activity(df: pd.DataFrame, out_dir: str = "results/figures",
                          filename: str = "session_activity_log.png"):
    """
    Scatter plot of session_duration vs. num_actions (log-log scale).
    Helps visualize both short and long sessions.
    """
    _ensure_dir(out_dir)

    action_cols = [c for c in df.columns if c.startswith("action_")]
    if not action_cols:
        print("⚠️ No action_* columns found. Skipping session activity plot.")
        return

    def _is_time_marker(x: Any) -> bool:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return False
        s = str(x)
        if not s.startswith("t"):
            return False
        return s[1:].replace(".", "").isdigit()

    session_duration, num_actions = [], []
    for row in df[action_cols].itertuples(index=False, name=None):
        dur = 0
        acts = 0
        for v in row:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            if _is_time_marker(v):
                dur += 1
            else:
                acts += 1
        if dur > 0 and acts > 0:
            session_duration.append(dur)
            num_actions.append(acts)

    if not session_duration:
        print("⚠️ No valid sessions found.")
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(session_duration, num_actions, alpha=0.3, s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Session Activity Patterns (log-log scale)")
    plt.xlabel("Session duration (log scale)")
    plt.ylabel("Number of actions (log scale)")
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"   ✓ Saved (log scale): {save_path}")

#====================================================
## Model Manager for Logistic Regression
#====================================================

class ModelManager:
    """
    Manages model saving, metrics tracking, and result organization.
    """
    
    def __init__(self, base_dir: str = "results"):
        """
        Initialize the ModelManager.
        
        Args:
            base_dir (str): Base directory for storing results
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.metrics_dir = os.path.join(base_dir, "metrics")
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def generate_model_name(self, model_type: str, timestamp: Optional[str] = None) -> str:
        """
        Generate a logical model name with timestamp.
        
        Args:
            model_type (str): Type of model (e.g., 'logistic_regression', 'random_forest')
            timestamp (str, optional): Custom timestamp, defaults to current time
            
        Returns:
            str: Generated model name
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{model_type}_{timestamp}"
    
    def save_model_with_metadata(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        scaler: Optional[Any] = None,
        feature_columns: Optional[list] = None,
        user_categories: Optional[Any] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model with comprehensive metadata.
        
        Args:
            model: The trained model object
            model_name: Name for the model
            model_type: Type of the model
            scaler: Fitted scaler object
            feature_columns: List of feature column names
            user_categories: User category mappings
            additional_metadata: Any additional metadata to store
            
        Returns:
            str: Path to the saved model
        """
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save main model
        model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
        joblib.dump(model, model_path)
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = os.path.join(model_dir, f"{model_name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
        
        # Save feature columns if provided
        if feature_columns is not None:
            features_path = os.path.join(model_dir, f"{model_name}_features.joblib")
            joblib.dump(feature_columns, features_path)
        
        # Save user categories if provided
        if user_categories is not None:
            categories_path = os.path.join(model_dir, f"{model_name}_categories.joblib")
            joblib.dump(user_categories, categories_path)
        
        # Create and save metadata
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "scaler_path": scaler_path if scaler is not None else None,
            "features_path": features_path if feature_columns is not None else None,
            "categories_path": categories_path if user_categories is not None else None,
            "additional_metadata": additional_metadata or {}
        }
        
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def save_metrics(
        self,
        model_name: str,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save metrics results in JSON format.
        
        Args:
            model_name: Name of the model (should match saved model)
            train_metrics: Training set metrics
            val_metrics: Validation set metrics
            test_metrics: Test set metrics (optional)
            additional_info: Any additional information to store
            
        Returns:
            str: Path to the saved metrics file
        """
        metrics_data = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics or {},
            "additional_info": additional_info or {}
        }
        
        # Save metrics file
        metrics_filename = f"{model_name}_metrics.json"
        metrics_path = os.path.join(self.metrics_dir, metrics_filename)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        return metrics_path
    
    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics if probabilities are available
        if y_pred_proba is not None:
            # Log loss
            from sklearn.metrics import log_loss
            try:
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except:
                metrics['log_loss'] = None
        
        return metrics
    
    def load_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Load metadata for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        metadata_path = os.path.join(self.models_dir, model_name, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found for model: {model_name}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_metrics(self, model_name: str) -> Dict[str, Any]:
        """
        Load metrics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model metrics
        """
        metrics_path = os.path.join(self.metrics_dir, f"{model_name}_metrics.json")
        
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Metrics not found for model: {model_name}")
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def list_models(self) -> list:
        """
        List all available models.
        
        Returns:
            list: List of model names
        """
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, f"{item}_metadata.json")
                if os.path.exists(metadata_path):
                    models.append(item)
        
        return sorted(models)
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a model including metadata and metrics.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Complete model summary
        """
        try:
            metadata = self.load_model_metadata(model_name)
            metrics = self.load_metrics(model_name)
            
            return {
                "model_name": model_name,
                "metadata": metadata,
                "metrics": metrics,
                "status": "complete"
            }
        except FileNotFoundError as e:
            return {
                "model_name": model_name,
                "error": str(e),
                "status": "incomplete"
            }
    
    def create_results_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all models and their metrics.
        
        Returns:
            pd.DataFrame: Summary of all models
        """
        models = self.list_models()
        summary_data = []
        
        for model_name in models:
            try:
                summary = self.get_model_summary(model_name)
                if summary["status"] == "complete":
                    row = {
                        "model_name": model_name,
                        "model_type": summary["metadata"]["model_type"],
                        "timestamp": summary["metadata"]["timestamp"],
                        "train_accuracy": summary["metrics"]["train_metrics"].get("accuracy", None),
                        "val_accuracy": summary["metrics"]["validation_metrics"].get("accuracy", None),
                        "train_f1_weighted": summary["metrics"]["train_metrics"].get("f1_weighted", None),
                        "val_f1_weighted": summary["metrics"]["validation_metrics"].get("f1_weighted", None),
                    }
                    summary_data.append(row)
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
        
        return pd.DataFrame(summary_data)


def save_model_results(
    model: Any,
    model_type: str,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    scaler: Optional[Any] = None,
    feature_columns: Optional[list] = None,
    user_categories: Optional[Any] = None,
    test_metrics: Optional[Dict[str, float]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Convenience function to save model and metrics in one call.
    
    Args:
        model: The trained model object
        model_type: Type of the model
        train_metrics: Training set metrics
        val_metrics: Validation set metrics
        scaler: Fitted scaler object
        feature_columns: List of feature column names
        user_categories: User category mappings
        test_metrics: Test set metrics (optional)
        additional_metadata: Any additional metadata to store
        
    Returns:
        Tuple[str, str]: (model_path, metrics_path)
    """
    manager = ModelManager()
    
    # Generate model name
    model_name = manager.generate_model_name(model_type)
    
    # Save model
    model_path = manager.save_model_with_metadata(
        model=model,
        model_name=model_name,
        model_type=model_type,
        scaler=scaler,
        feature_columns=feature_columns,
        user_categories=user_categories,
        additional_metadata=additional_metadata
    )
    
    # Save metrics
    metrics_path = manager.save_metrics(
        model_name=model_name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        additional_info=additional_metadata
    )
    
    return model_path, metrics_path

def main_viz(data_dir: str = "data", size: str = "full", out_dir: str = "results/figures"):
    print("=" * 80)
    print("Copilote EDA Visualizations")
    print("=" * 80)

    print("\n[1/3] Loading training data...")
    df_train = load_data("train", data_dir=data_dir, size=size)

    print("\n[2/3] Cleaning data...")
    df_train_clean = clean_data(df_train, is_train=True)

    print("\n[3/3] Creating figures...")
    # plot_class_distribution(df_train_clean, out_dir=out_dir, filename="class_balance.png")
    # plot_browser_usage(df_train_clean, out_dir=out_dir, filename="browser_usage.png")
    plot_session_activity(df_train_clean, out_dir=out_dir, filename="session_activity.png")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate EDA visualizations for Copilote dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--size", type=str, default="full", choices=["full", "small"],
                        help="Subset size for quick runs")
    parser.add_argument("--out_dir", type=str, default="results/figures", help="Output directory for figures")
    args = parser.parse_args()
    main_viz(data_dir=args.data_dir, size=args.size, out_dir=args.out_dir)
