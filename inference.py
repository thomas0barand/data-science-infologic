"""
Inference script for Attention-LSTM User Identification

This script generates predictions on test data using a trained model checkpoint.

Usage:
    python inference.py --checkpoint results/rnn_models/attention_lstm_light_20251020_003801/best.ckpt
    python inference.py --checkpoint results/rnn_models/attention_lstm_light_20251020_003801/best.ckpt --output my_submission.csv
    python inference.py -c results/rnn_models/attention_lstm_light_20251020_003801/best.ckpt
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import joblib
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import (
    load_data, clean_data, tokenize_actions, 
    prepare_rnn_sequences, extract_statistical_features
)
from rnn_attention_lightning import AttentionLSTMClassifier


def load_model_and_artifacts(checkpoint_path):
    """
    Load trained model and required artifacts (vocabulary, user_categories, scaler).
    
    Args:
        checkpoint_path (str): Path to model checkpoint (.ckpt file)
        
    Returns:
        tuple: (model, vocabulary, user_categories, scaler, config)
    """
    print("\n" + "=" * 80)
    print("Loading Model and Artifacts")
    print("=" * 80)
    
    # Get model directory (checkpoint file's parent directory)
    model_dir = os.path.dirname(checkpoint_path)
    artifacts_dir = os.path.join(model_dir, 'artifacts')
    config_path = os.path.join(model_dir, 'config.yaml')
    
    # Check if paths exist
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Artifacts directory: {artifacts_dir}")
    print(f"✓ Config file: {config_path}")
    
    # Load artifacts
    print("\n📦 Loading artifacts...")
    vocabulary = joblib.load(os.path.join(artifacts_dir, 'vocabulary.joblib'))
    user_categories = joblib.load(os.path.join(artifacts_dir, 'user_categories.joblib'))
    scaler = joblib.load(os.path.join(artifacts_dir, 'stat_scaler.joblib'))
    
    print(f"  • Vocabulary size: {len(vocabulary)}")
    print(f"  • Number of users: {len(user_categories)}")
    print(f"  • Statistical feature scaler: {scaler.__class__.__name__}")
    
    # Load config
    print("\n🔧 Loading configuration...")
    config = OmegaConf.load(config_path)
    print(f"  • Model type: Attention-LSTM")
    print(f"  • Embedding dim: {config.model.embedding_dim}")
    print(f"  • Hidden size: {config.model.hidden_size}")
    print(f"  • Bidirectional: {config.model.bidirectional}")
    
    # Load checkpoint to get hyperparameters
    print("\n🔍 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get dimensions from checkpoint state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        
        # Get actual vocab_size from embedding layer
        embedding_weight = state_dict.get('embedding.weight')
        vocab_size = embedding_weight.shape[0] if embedding_weight is not None else len(vocabulary)
        
        # Get actual num_users from classifier layer (this is the ground truth)
        classifier_weight = state_dict.get('classifier.weight')
        if classifier_weight is not None:
            num_users = classifier_weight.shape[0]
        else:
            num_users = len(user_categories)
        
        # Get stat_feature_dim from stat_bn layer
        stat_bn_weight = state_dict.get('stat_bn.weight')
        if stat_bn_weight is not None:
            stat_feature_dim = stat_bn_weight.shape[0]
        else:
            # Fallback: try to infer from fusion layer
            fusion_weight = state_dict.get('fusion.0.weight')
            if fusion_weight is not None:
                # fusion input = lstm_output + stat_features + browser_features (4)
                lstm_output_size = config.model.hidden_size * (2 if config.model.bidirectional else 1)
                browser_dim = 4
                stat_feature_dim = fusion_weight.shape[1] - lstm_output_size - browser_dim
            else:
                print("⚠️  Warning: Could not infer stat_feature_dim from checkpoint, using default 14")
                stat_feature_dim = 14  # Default value
    else:
        # Fallback to artifacts
        vocab_size = len(vocabulary)
        num_users = len(user_categories)
        stat_feature_dim = 14
    
    print(f"  • Vocab size: {vocab_size}")
    print(f"  • Num users (from checkpoint): {num_users}")
    print(f"  • Num users (from categories): {len(user_categories)}")
    print(f"  • Stat feature dim: {stat_feature_dim}")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = AttentionLSTMClassifier(
        config=config,
        vocab_size=vocab_size,
        num_users=num_users,
        stat_feature_dim=stat_feature_dim
    )
    
    # Load state dict (filter out criterion weights if present)
    state_dict = checkpoint['state_dict']
    # Remove criterion weights from state dict (they're not part of the model architecture)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('criterion.')}
    model.load_state_dict(filtered_state_dict)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  • Total parameters: {total_params:,}")
    
    print("\n✓ Model and artifacts loaded successfully!")
    
    return model, vocabulary, user_categories, scaler, config


def process_test_data(test_csv_path, vocabulary, scaler, config):
    """
    Process test data for inference.
    
    Args:
        test_csv_path (str): Path to test.csv
        vocabulary (dict): Vocabulary from training
        scaler: Statistical feature scaler from training
        config: Model configuration
        
    Returns:
        tuple: (X_sequences, X_stat, X_browser)
    """
    print("\n" + "=" * 80)
    print("Processing Test Data")
    print("=" * 80)
    
    # Load test data
    print("\n📂 Loading test data...")
    df_test = load_data('test', data_dir=os.path.dirname(test_csv_path), size='full')
    print(f"✓ Loaded {len(df_test)} test samples")
    
    # Clean data
    print("\n🧹 Cleaning data...")
    df_test_clean = clean_data(df_test, is_train=False)
    print(f"✓ Cleaned test data: {len(df_test_clean)} samples")
    
    # Tokenize sequences
    print("\n🔤 Tokenizing sequences...")
    sequences, _ = tokenize_actions(df_test_clean, is_train=False, vocabulary=vocabulary)
    print(f"✓ Tokenized {len(sequences)} sequences")
    print(f"  • Average sequence length: {np.mean([len(s) for s in sequences]):.1f} actions")
    
    # Prepare RNN sequences
    print("\n📊 Preparing RNN sequences...")
    X_sequences, X_browser, _, _ = prepare_rnn_sequences(
        df_test_clean, vocabulary, 
        max_length=config.data.max_sequence_length,
        is_train=False
    )
    print(f"✓ Sequences shape: {X_sequences.shape}")
    print(f"✓ Browser features shape: {X_browser.shape}")
    
    # Extract statistical features
    print("\n📈 Extracting statistical features...")
    X_stat = extract_statistical_features(df_test_clean, action_sequences=sequences)
    print(f"✓ Statistical features shape: {X_stat.shape}")
    
    # Normalize statistical features using training scaler
    print("\n🔄 Normalizing statistical features...")
    X_stat_normalized = scaler.transform(X_stat)
    print(f"✓ Statistical features normalized")
    
    return X_sequences, X_stat_normalized, X_browser


def generate_predictions(model, X_sequences, X_stat, X_browser, user_categories, 
                        batch_size=64, device='cpu'):
    """
    Generate predictions for test data.
    
    Args:
        model: Trained model
        X_sequences (np.ndarray): Action sequences
        X_stat (np.ndarray): Statistical features (normalized)
        X_browser (np.ndarray): Browser features
        user_categories: Mapping from category index to user_id
        batch_size (int): Batch size for inference
        device (str): Device to run inference on
        
    Returns:
        np.ndarray: Predicted user IDs
    """
    print("\n" + "=" * 80)
    print("Generating Predictions")
    print("=" * 80)
    
    model.to(device)
    model.eval()
    
    num_samples = len(X_sequences)
    all_predictions = []
    
    print(f"\n🔮 Running inference on {num_samples} samples...")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Device: {device}")
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Inference"):
            # Get batch
            batch_end = min(i + batch_size, num_samples)
            batch_sequences = torch.LongTensor(X_sequences[i:batch_end]).to(device)
            batch_stat = torch.FloatTensor(X_stat[i:batch_end]).to(device)
            batch_browser = torch.FloatTensor(X_browser[i:batch_end]).to(device)
            
            # Get predictions (category indices)
            batch_predictions = model.predict(batch_sequences, batch_stat, batch_browser)
            all_predictions.extend(batch_predictions)
    
    predictions_array = np.array(all_predictions)
    print(f"\n✓ Generated {len(predictions_array)} predictions")
    
    # Map category indices to actual user IDs
    print("\n🔍 Mapping predictions to user IDs...")
    predicted_user_ids = []
    
    # Check if user_categories is a dict, pandas Categorical, LabelEncoder, or similar
    if isinstance(user_categories, dict):
        # user_categories is a dict: category_idx -> user_id
        for pred_idx in predictions_array:
            user_id = user_categories.get(pred_idx, f"unknown_user_{pred_idx}")
            predicted_user_ids.append(user_id)
    else:
        # Try different approaches based on object type
        try:
            # Check if it's a pandas Categorical
            if hasattr(user_categories, 'categories'):
                # It's a pandas Categorical - use the categories directly
                categories = user_categories.categories
                predicted_user_ids = [categories[idx] for idx in predictions_array]
            elif hasattr(user_categories, 'inverse_transform'):
                # It's a LabelEncoder or similar
                predicted_user_ids = user_categories.inverse_transform(predictions_array)
            elif hasattr(user_categories, '__getitem__'):
                # It's array-like - index directly
                predicted_user_ids = [user_categories[idx] for idx in predictions_array]
            else:
                # Fallback: use predictions as-is
                print("⚠️  Warning: Could not map predictions, using indices as user IDs")
                predicted_user_ids = [f"user_{pred_idx}" for pred_idx in predictions_array]
        except Exception as e:
            # Fallback: use predictions as-is
            print(f"⚠️  Warning: Could not map predictions ({e}), using indices as user IDs")
            predicted_user_ids = [f"user_{pred_idx}" for pred_idx in predictions_array]
    
    print(f"✓ Mapped predictions to user IDs")
    print(f"  • Unique predicted users: {len(set(predicted_user_ids))}")
    
    # Show sample predictions
    if len(predicted_user_ids) > 0:
        sample_predictions = list(predicted_user_ids[:10])
        print(f"  • Sample predictions: {sample_predictions}")
    
    return predicted_user_ids


def create_submission_file(predictions, output_path='submission.csv'):
    """
    Create submission CSV file in the required format.
    
    Args:
        predictions (list): List of predicted user IDs
        output_path (str): Path to save submission file
    """
    print("\n" + "=" * 80)
    print("Creating Submission File")
    print("=" * 80)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'RowId': range(1, len(predictions) + 1),
        'prediction': predictions
    })
    
    print(f"\n📝 Submission format:")
    print(submission_df.head(10))
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"\n✓ Submission file saved to: {output_path}")
    print(f"  • Total predictions: {len(predictions)}")
    print(f"  • File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    return submission_df


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Generate predictions using trained Attention-LSTM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --checkpoint results/rnn_models/attention_lstm_light_20251020_003801/best.ckpt
  python inference.py -c results/rnn_models/model_name/best.ckpt --output my_submission.csv
  python inference.py -c results/rnn_models/model_name/last.ckpt --device cuda
        """
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    
    parser.add_argument(
        '--test-data', '-t',
        type=str,
        default='data/test.csv',
        help='Path to test.csv (default: data/test.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='submission.csv',
        help='Output submission file path (default: submission.csv)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size for inference (default: 64)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to run inference on (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("Attention-LSTM User Identification - Inference")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"  • Checkpoint: {args.checkpoint}")
    print(f"  • Test data: {args.test_data}")
    print(f"  • Output file: {args.output}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Device: {args.device}")
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"  • Using device: {device}")
    
    # Step 1: Load model and artifacts
    model, vocabulary, user_categories, scaler, config = load_model_and_artifacts(
        args.checkpoint
    )
    
    # Step 2: Process test data
    X_sequences, X_stat, X_browser = process_test_data(
        args.test_data, vocabulary, scaler, config
    )
    
    # Step 3: Generate predictions
    predictions = generate_predictions(
        model, X_sequences, X_stat, X_browser, user_categories,
        batch_size=args.batch_size, device=device
    )
    
    # Step 4: Create submission file
    submission_df = create_submission_file(predictions, args.output)
    
    # Final summary
    print("\n" + "=" * 80)
    print("Inference Complete! 🎉")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"  • Test samples processed: {len(predictions)}")
    print(f"  • Unique users predicted: {len(set(predictions))}")
    print(f"  • Submission file: {args.output}")
    print("\n✨ Your submission is ready for upload!")
    print("=" * 80)


if __name__ == "__main__":
    main()

