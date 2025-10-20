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


def reconstruct_config_from_checkpoint(checkpoint):
    """
    Reconstruct configuration from checkpoint when config.yaml is missing.
    
    Args:
        checkpoint (dict): PyTorch Lightning checkpoint
        
    Returns:
        OmegaConf: Reconstructed configuration
    """
    state_dict = checkpoint['state_dict']
    
    # Extract architecture parameters from state_dict
    embedding_weight = state_dict.get('embedding.weight')
    lstm_weight_ih_l0 = state_dict.get('lstm.weight_ih_l0')
    stat_bn_weight = state_dict.get('stat_bn.weight')
    fusion_weight = state_dict.get('fusion.0.weight')
    classifier_weight = state_dict.get('classifier.weight')
    
    # Infer parameters
    vocab_size = embedding_weight.shape[0] if embedding_weight is not None else 4000
    embedding_dim = embedding_weight.shape[1] if embedding_weight is not None else 64
    
    # LSTM hidden size - weight_ih has shape (4*hidden_size, input_size)
    if lstm_weight_ih_l0 is not None:
        hidden_size = lstm_weight_ih_l0.shape[0] // 4
    else:
        hidden_size = 128
    
    # Check if bidirectional by looking for reverse layer weights
    bidirectional = 'lstm.weight_ih_l0_reverse' in state_dict
    
    # Get number of layers
    num_layers = 1
    for key in state_dict.keys():
        if 'lstm.weight_ih_l' in key:
            layer_num = int(key.split('_l')[1].split('_')[0].replace('reverse', ''))
            num_layers = max(num_layers, layer_num + 1)
    
    # Statistical feature dimension
    stat_feature_dim = stat_bn_weight.shape[0] if stat_bn_weight is not None else 22
    
    # Number of users (output classes)
    num_users = classifier_weight.shape[0] if classifier_weight is not None else 247
    
    # Attention dimension - infer from attention layer
    attention_weight = state_dict.get('attention.W.weight')
    if attention_weight is not None:
        attention_dim = attention_weight.shape[0]
    else:
        attention_dim = hidden_size * (2 if bidirectional else 1)
    
    # Fusion hidden size
    if fusion_weight is not None:
        fusion_hidden_size = fusion_weight.shape[0]
    else:
        fusion_hidden_size = 256
    
    # Create minimal config with all required sections
    config_dict = {
        'model': {
            'embedding_dim': int(embedding_dim),
            'hidden_size': int(hidden_size),
            'num_layers': int(num_layers),
            'dropout': 0.3,  # Default
            'bidirectional': bool(bidirectional),
            'attention_dim': int(attention_dim),
            'fusion_hidden_size': int(fusion_hidden_size)
        },
        'data': {
            'max_sequence_length': 2000,  # Default
            'batch_size': 64
        },
        'training': {
            'use_focal_loss': False,  # Default for inference
            'focal_loss': {
                'gamma': 2.0,
                'alpha_mode': 'balanced'
            }
        }
    }
    
    config = OmegaConf.create(config_dict)
    
    return config


def load_model_and_artifacts(checkpoint_path):
    """
    Load trained model and required artifacts (vocabulary, user_categories, scaler).
    
    Config file is optional - the script can reconstruct model architecture from checkpoint.
    
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
    
    # Artifacts directory is optional - we can work without it if vocabulary/scaler are found elsewhere
    has_artifacts_dir = os.path.exists(artifacts_dir)
    has_config_file = os.path.exists(config_path)
    
    print(f"✓ Checkpoint: {checkpoint_path}")
    if has_artifacts_dir:
        print(f"✓ Artifacts directory: {artifacts_dir}")
    else:
        print(f"⚠️  Artifacts directory not found (will try to locate artifacts elsewhere)")
    if has_config_file:
        print(f"✓ Config file: {config_path}")
    else:
        print(f"⚠️  Config file not found (will reconstruct from checkpoint)")
    
    # Load checkpoint first to check what's available
    print("\n🔍 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if hyperparameters are in checkpoint
    has_hyperparams = 'hyper_parameters' in checkpoint
    if has_hyperparams:
        hparams = checkpoint['hyper_parameters']
        print(f"✓ Found hyperparameters in checkpoint")
    else:
        print(f"⚠️  No hyperparameters in checkpoint, will need config file")
    
    # Load artifacts (try multiple locations)
    print("\n📦 Loading artifacts...")
    
    # Try to load vocabulary (search in multiple locations)
    vocabulary = None
    vocab_paths = [
        os.path.join(artifacts_dir, 'vocabulary.joblib') if has_artifacts_dir else None,
        # Fallback: search in results/rnn_models for any vocabulary.joblib
        *[os.path.join(root, 'artifacts', 'vocabulary.joblib') 
          for root, dirs, files in os.walk('results/rnn_models') 
          if 'artifacts' in dirs][:3]  # Check up to 3 other models
    ]
    
    for vocab_path in vocab_paths:
        if vocab_path and os.path.exists(vocab_path):
            try:
                vocabulary = joblib.load(vocab_path)
                print(f"  ✓ Vocabulary loaded: {len(vocabulary)} tokens")
                if vocab_path != vocab_paths[0]:
                    print(f"    (from fallback: {vocab_path})")
                break
            except:
                continue
    
    if vocabulary is None:
        print(f"  ⚠️  Vocabulary not found, will build from test data (predictions may be poor)")
    
    # Try to load user_categories (search in multiple locations)
    user_categories = None
    user_cat_paths = [
        os.path.join(artifacts_dir, 'user_categories.joblib') if has_artifacts_dir else None,
        # Fallback: search in results/rnn_models
        *[os.path.join(root, 'artifacts', 'user_categories.joblib') 
          for root, dirs, files in os.walk('results/rnn_models') 
          if 'artifacts' in dirs][:3]
    ]
    
    for user_cat_path in user_cat_paths:
        if user_cat_path and os.path.exists(user_cat_path):
            try:
                user_categories = joblib.load(user_cat_path)
                if hasattr(user_categories, 'categories'):
                    print(f"  ✓ User categories loaded: {len(user_categories.categories)} users")
                elif hasattr(user_categories, '__len__'):
                    print(f"  ✓ User categories loaded: {len(user_categories)} users")
                else:
                    print(f"  ✓ User categories loaded")
                if user_cat_path != user_cat_paths[0]:
                    print(f"    (from fallback: {user_cat_path})")
                break
            except:
                continue
    
    if user_categories is None:
        print(f"  ⚠️  User categories not found, predictions will use indices")
    
    # Try to load scaler (search in multiple locations)
    scaler = None
    scaler_paths = [
        os.path.join(artifacts_dir, 'stat_scaler.joblib') if has_artifacts_dir else None,
        # Fallback: search in results/rnn_models
        *[os.path.join(root, 'artifacts', 'stat_scaler.joblib') 
          for root, dirs, files in os.walk('results/rnn_models') 
          if 'artifacts' in dirs][:3]
    ]
    
    for scaler_path in scaler_paths:
        if scaler_path and os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                print(f"  ✓ Statistical feature scaler: {scaler.__class__.__name__}")
                if scaler_path != scaler_paths[0]:
                    print(f"    (from fallback: {scaler_path})")
                break
            except:
                continue
    
    if scaler is None:
        print(f"  ⚠️  Scaler not found, will use raw features (predictions may be degraded)")
    
    # Load or reconstruct config
    print("\n🔧 Loading configuration...")
    if has_config_file:
        config = OmegaConf.load(config_path)
        print(f"  ✓ Config loaded from file")
    elif has_hyperparams and 'state_dict' in checkpoint:
        # Reconstruct config from checkpoint
        print(f"  🔨 Reconstructing config from checkpoint...")
        config = reconstruct_config_from_checkpoint(checkpoint)
        print(f"  ✓ Config reconstructed from checkpoint")
    else:
        raise RuntimeError("Cannot load or reconstruct config. Need either config.yaml or checkpoint with hyperparameters")
    
    print(f"  • Model type: Attention-LSTM")
    print(f"  • Embedding dim: {config.model.embedding_dim}")
    print(f"  • Hidden size: {config.model.hidden_size}")
    print(f"  • Bidirectional: {config.model.bidirectional}")
    
    # Get dimensions from checkpoint state_dict
    print("\n📐 Extracting model dimensions...")
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
    if user_categories is not None:
        if hasattr(user_categories, 'categories'):
            print(f"  • Num users (from categories): {len(user_categories.categories)}")
        elif hasattr(user_categories, '__len__'):
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
        vocabulary (dict or None): Vocabulary from training (None = build from test data)
        scaler (StandardScaler or None): Statistical feature scaler (None = use raw features)
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
    if vocabulary is None:
        print("  ⚠️  No vocabulary provided, building from test data (may miss training tokens)")
        sequences, vocabulary = tokenize_actions(df_test_clean, is_train=True)  # Build vocab
    else:
        sequences, _ = tokenize_actions(df_test_clean, is_train=False, vocabulary=vocabulary)
    
    print(f"✓ Tokenized {len(sequences)} sequences")
    print(f"  • Vocabulary size: {len(vocabulary)}")
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
    if scaler is not None:
        print("\n🔄 Normalizing statistical features...")
        X_stat_normalized = scaler.transform(X_stat)
        print(f"✓ Statistical features normalized")
    else:
        print("\n⚠️  No scaler provided, using raw statistical features")
        X_stat_normalized = X_stat
    
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

