"""
Evaluate a checkpoint on the training dataset and compute metrics.
Optionally generate submission.csv from test data.

This script:
1. Loads a checkpoint (reconstructs config if missing)
2. Runs inference on the train dataset and computes metrics
3. Optionally generates submission.csv from test data

The metrics JSON follows the same format as the training script output.

Usage:
    # Evaluate on train dataset only
    python evaluate_checkpoint.py --checkpoint results/models/attention_lstm_light_20251020_102622/epoch=epoch=149-val_f1=val_f1=0.8628.ckpt
    
    # Evaluate on train dataset AND generate submission.csv from test data
    python evaluate_checkpoint.py --checkpoint results/models/attention_lstm_light_20251020_102622/epoch=epoch=149-val_f1=val_f1=0.8628.ckpt --test-path test --submission submission.csv
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
import joblib
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from utils.utils import (
    load_data, clean_data, tokenize_actions, 
    prepare_rnn_sequences, extract_statistical_features
)
from model.attention_lstm import AttentionLSTMClassifier
from inference_attention_lstm import (
    load_model_and_artifacts,
    prepare_test_data,
    generate_predictions,
    reconstruct_config_from_checkpoint
)


def prepare_train_data(train_df, vocabulary, scaler, max_length):
    """
    Prepare train data for prediction (similar to test data preparation).
    
    Args:
        train_df (pd.DataFrame): Training dataframe with user_id column
        vocabulary (dict): Action vocabulary
        scaler: Statistical feature scaler
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (sequences, browser_features, stat_features, true_labels)
            where true_labels are the encoded user IDs
    """
    print(f"\n🔧 Preparing train data...")
    
    # Tokenize and prepare sequences (needed for statistical features)
    action_sequences, _ = tokenize_actions(train_df, is_train=False, vocabulary=vocabulary)
    
    # Prepare RNN sequences with browser features
    sequences, browser_features, _, _ = prepare_rnn_sequences(
        train_df, vocabulary, max_length=max_length, is_train=False
    )
    
    # Extract statistical features
    stat_features = extract_statistical_features(train_df, action_sequences=action_sequences)
    
    # Normalize statistical features using training scaler
    stat_features_normalized = scaler.transform(stat_features)
    
    # Get true labels (user_id) - convert to categorical codes
    user_id_cat = pd.Categorical(train_df['user_id'])
    true_labels = user_id_cat.codes
    
    print(f"✓ Train sequences shape: {sequences.shape}")
    print(f"✓ Browser features shape: {browser_features.shape}")
    print(f"✓ Statistical features shape: {stat_features_normalized.shape}")
    print(f"✓ True labels shape: {true_labels.shape}")
    print(f"✓ Number of unique users: {len(np.unique(true_labels))}")
    
    return sequences, browser_features, stat_features_normalized, true_labels


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive metrics from predictions.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary of computed metrics matching the reference format
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    return metrics


def save_metrics(metrics, config, vocab_size, num_users, checkpoint_path, model_name=None, output_path=None):
    """
    Save metrics to JSON file in the same format as training script.
    
    Args:
        metrics (dict): Metrics dictionary (train metrics)
        config: OmegaConf configuration object
        vocab_size (int): Vocabulary size
        num_users (int): Number of users
        checkpoint_path (str): Path to checkpoint
        model_name (str, optional): Model name (will be inferred if not provided)
        output_path (str, optional): Output path for metrics JSON
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Infer model name from checkpoint directory if not provided
    if model_name is None:
        model_dir_name = os.path.basename(checkpoint_dir)
        model_name = model_dir_name
    
    if output_path is None:
        # Create metrics filename based on model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/metrics/{model_name}_{timestamp}_metrics.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert config to dictionary (resolving values)
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Prepare metrics data matching the reference format
    metrics_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'config': config_dict,
        'data': {
            'vocab_size': vocab_size,
            'num_users': num_users,
            'max_sequence_length': config.data.max_sequence_length,
            'data_size': config.data.get('size', 'full')
        },
        'architecture': {
            'model_type': 'Attention-LSTM',
            'embedding_dim': config.model.embedding_dim,
            'hidden_size': config.model.hidden_size,
            'num_layers': config.model.num_layers,
            'dropout': config.model.dropout,
            'bidirectional': config.model.bidirectional,
            'attention_dim': config.model.attention_dim,
            'fusion_hidden_size': config.model.get('fusion_hidden_size', 512),
            'use_focal_loss': config.training.get('use_focal_loss', False)
        },
        'training': {
            'optimizer': config.optimizer.name,
            'learning_rate': config.optimizer.lr,
            'weight_decay': config.optimizer.get('weight_decay', 0.0001),
            'scheduler': config.scheduler.name,
            'max_epochs': config.training.get('max_epochs', 200),
            'batch_size': config.data.batch_size
        },
        'metrics': {
            'train': metrics
        },
        'checkpoint': {
            'best_model_path': checkpoint_path
        }
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\n💾 Metrics saved to: {output_path}")
    
    return output_path


def save_submission(predictions, user_categories, output_path="submission.csv"):
    """
    Save predictions to submission file in the format of sample_submission.csv.
    
    Args:
        predictions (np.ndarray): Predicted user codes
        user_categories: User category mapping (pandas Categorical)
        output_path (str): Output file path
    """
    print(f"\n💾 Saving submission file...")
    
    # Map codes back to user IDs
    # Based on inference script, user_categories.categories[0] contains the actual categories
    try:
        if hasattr(user_categories, 'categories'):
            # Handle pandas Categorical - categories[0] is the actual index
            if isinstance(user_categories.categories, (list, tuple)) and len(user_categories.categories) > 0:
                cats = user_categories.categories[0]
                user_ids = np.array([cats[p] if p < len(cats) else f"user_{p}" for p in predictions])
            elif isinstance(user_categories.categories, (pd.Index, np.ndarray)):
                cats = user_categories.categories
                user_ids = np.array([cats[p] if p < len(cats) else f"user_{p}" for p in predictions])
            else:
                # Try direct indexing
                user_ids = user_categories.categories[predictions]
        else:
            # Fallback: assume it's already the mapping array
            user_ids = user_categories[predictions] if hasattr(user_categories, '__getitem__') else predictions
    except (TypeError, IndexError, AttributeError) as e:
        print(f"⚠️  Standard categorical indexing failed: {e}")
        print(f"   Trying alternative approach...")
        
        # Alternative: try to extract categories directly
        if hasattr(user_categories, 'categories'):
            cats = user_categories.categories
            # Handle nested structure
            if isinstance(cats, (list, tuple)) and len(cats) > 0:
                if isinstance(cats[0], (list, tuple, pd.Index, np.ndarray)):
                    cats = cats[0]
                else:
                    cats = cats
            user_ids = np.array([cats[p] if p < len(cats) else f"user_{p}" for p in predictions])
        else:
            # Last resort: use predictions directly
            user_ids = predictions
    
    # Create submission dataframe matching sample_submission.csv format
    # Format: RowId,prediction
    submission_df = pd.DataFrame({
        'RowId': range(1, len(user_ids) + 1),
        'prediction': user_ids
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"✓ Submission saved to: {output_path}")
    print(f"  • Number of predictions: {len(submission_df)}")
    print(f"  • Unique users predicted: {submission_df['prediction'].nunique()}")
    
    # Show distribution
    value_counts = submission_df['prediction'].value_counts()
    print(f"  • Most frequent user: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
    print(f"  • Distribution: min={value_counts.min()}, max={value_counts.max()}, "
          f"median={value_counts.median():.1f}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate checkpoint on training dataset and compute metrics. Optionally generate submission.csv from test data.'
    )
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--train-path', type=str, default='train',
                       help='Training data path (default: train)')
    parser.add_argument('--test-path', type=str, default=None,
                       help='Test data path for generating submission.csv (optional)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference (default: 64)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for metrics JSON (default: auto-generated)')
    parser.add_argument('--submission', '-s', type=str, default=None,
                       help='Output path for submission.csv (only used if --test-path is provided, default: submission.csv)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Attention-LSTM Checkpoint Evaluation")
    print("=" * 80)
    
    # Load model and artifacts (will reconstruct config if missing)
    print(f"\n📦 Loading model from checkpoint...")
    model, vocabulary, user_categories, scaler, config = load_model_and_artifacts(
        args.checkpoint,
        train_data_path=args.train_path,
        data_dir=args.data_dir
    )
    
    # Get vocab_size and num_users for metrics
    vocab_size = len(vocabulary)
    if hasattr(user_categories, 'categories'):
        if isinstance(user_categories.categories, (list, tuple)) and len(user_categories.categories) > 0:
            # Handle nested categories
            if isinstance(user_categories.categories[0], (list, tuple, pd.Index, np.ndarray)):
                num_users = len(user_categories.categories[0])
            else:
                num_users = len(user_categories.categories)
        elif isinstance(user_categories.categories, (pd.Index, np.ndarray)):
            num_users = len(user_categories.categories)
        else:
            num_users = len(set(user_categories))
    else:
        num_users = len(set(user_categories))
    
    # Load train data for evaluation
    print(f"\n📂 Loading training data...")
    print(f"   Train path: {args.train_path}")
    print(f"   Data directory: {args.data_dir}")
    
    train_df = load_data(args.train_path, data_dir=args.data_dir, size='full')
    print(f"✓ Loaded {len(train_df)} training samples")
    
    # Clean train data
    train_df_clean = clean_data(train_df, is_train=True)
    
    # Prepare train data (sequences + browser features + statistical features + true labels)
    max_length = config.data.max_sequence_length
    sequences, browser_features, stat_features, true_labels = prepare_train_data(
        train_df_clean, vocabulary, scaler, max_length
    )
    
    # Generate predictions on train data
    print(f"\n🔮 Generating predictions on training data...")
    train_predictions = generate_predictions(
        model, sequences, stat_features, browser_features, 
        batch_size=args.batch_size
    )
    
    # Compute metrics
    print(f"\n📊 Computing metrics...")
    train_metrics = compute_metrics(true_labels, train_predictions)
    
    # Print metrics
    print(f"\n📈 Training Dataset Metrics:")
    print(f"  • Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  • F1 Score (macro): {train_metrics['f1_macro']:.4f}")
    print(f"  • F1 Score (weighted): {train_metrics['f1_weighted']:.4f}")
    print(f"  • Precision (weighted): {train_metrics['precision']:.4f}")
    print(f"  • Recall (weighted): {train_metrics['recall']:.4f}")
    
    # Infer model name from checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint)
    model_name = os.path.basename(checkpoint_dir)
    
    # Save metrics
    metrics_path = save_metrics(
        train_metrics, config, vocab_size, num_users, args.checkpoint,
        model_name=model_name, output_path=args.output
    )
    
    # Optionally generate submission.csv from test data
    if args.test_path:
        print(f"\n📂 Loading test data for submission generation...")
        print(f"   Test path: {args.test_path}")
        
        test_df = load_data(args.test_path, data_dir=args.data_dir, size='full')
        print(f"✓ Loaded {len(test_df)} test samples")
        
        # Clean test data
        test_df_clean = clean_data(test_df, is_train=False)
        
        # Prepare test data (sequences + browser features + statistical features)
        test_sequences, test_browser_features, test_stat_features = prepare_test_data(
            test_df_clean, vocabulary, scaler, max_length
        )
        
        # Generate predictions on test data
        print(f"\n🔮 Generating predictions on test data...")
        test_predictions = generate_predictions(
            model, test_sequences, test_stat_features, test_browser_features,
            batch_size=args.batch_size
        )
        
        # Save submission
        submission_path = args.submission if args.submission else "submission.csv"
        save_submission(test_predictions, user_categories, output_path=submission_path)
    
    print("\n" + "=" * 80)
    print("Evaluation completed! 🎉")
    print("=" * 80)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    if args.test_path:
        submission_path = args.submission if args.submission else "submission.csv"
        print(f"✓ Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()

