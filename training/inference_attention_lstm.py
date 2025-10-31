"""
Inference script for Attention-LSTM User Identification

This script generates predictions on test data using a trained Attention-LSTM model.

Usage:
    python inference_attention.py --checkpoint results/rnn_models/attention_lstm_light_20251020_010507/epoch=epoch=75-val_f1=val_f1=0.8636.ckpt
    python inference_attention.py --checkpoint results/rnn_models/attention_lstm_light_20251020_010507/epoch=epoch=75-val_f1=val_f1=0.8636.ckpt --output my_submission.csv
    python inference_attention.py -c results/rnn_models/attention_lstm_light_20251020_010507/epoch=epoch=75-val_f1=val_f1=0.8636.ckpt
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.utils import (
    load_data, clean_data, tokenize_actions, 
    prepare_rnn_sequences, extract_statistical_features
)
from model.attention_lstm import AttentionLSTMClassifier
# from rnn_attention_simple import AttentionLSTMClassifierSimple


def regenerate_artifacts(train_path, data_dir):
    """
    Regenerate vocabulary, user categories, and scaler from training data.
    
    This follows the same preprocessing pipeline as in training.
    
    Args:
        train_path (str): Path to training data
        data_dir (str): Data directory
        
    Returns:
        tuple: (vocabulary, user_categories, scaler)
    """
    print(f"\n🔧 Regenerating artifacts from training data...")
    
    # Load and clean training data
    df_train = load_data(train_path, data_dir=data_dir, size='full')
    print(f"✓ Loaded {len(df_train)} training samples")
    
    df_train_clean = clean_data(df_train, is_train=True)
    
    # Build vocabulary
    sequences, vocabulary = tokenize_actions(df_train_clean, is_train=True)
    print(f"✓ Built vocabulary: {len(vocabulary)} tokens")
    
    # Get user categories (same format as in training)
    import pandas as pd
    user_id_cat = pd.Categorical(df_train_clean['user_id'])
    user_categories = user_id_cat
    print(f"✓ Extracted user categories: {len(user_categories.categories)} users")
    
    # Extract statistical features for scaler
    stat_features = extract_statistical_features(df_train_clean, action_sequences=sequences)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(stat_features)
    print(f"✓ Fitted statistical feature scaler: {stat_features.shape[1]} features")
    
    return vocabulary, user_categories, scaler


def load_model_and_artifacts(checkpoint_path, train_data_path=None, data_dir='data'):
    """
    Load trained model and required artifacts.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        train_data_path (str, optional): Path to training data (if artifacts need to be regenerated)
        data_dir (str): Data directory
        
    Returns:
        tuple: (model, vocabulary, user_categories, scaler, config)
    """
    print(f"\n📦 Loading model and artifacts from checkpoint...")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Get directory containing checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    artifacts_dir = os.path.join(checkpoint_dir, 'artifacts')
    
    # Load vocabulary and user categories
    vocab_path = os.path.join(artifacts_dir, 'vocabulary.joblib')
    user_cat_path = os.path.join(artifacts_dir, 'user_categories.joblib')
    scaler_path = os.path.join(artifacts_dir, 'stat_scaler.joblib')
    
    # Check if artifacts exist, if not regenerate from training data
    artifacts_exist = all([
        os.path.exists(vocab_path),
        os.path.exists(user_cat_path),
        os.path.exists(scaler_path)
    ])
    
    if not artifacts_exist:
        print(f"⚠️  Artifacts not found in {artifacts_dir}")
        if train_data_path:
            print(f"🔄 Regenerating artifacts from training data...")
            vocabulary, user_categories, scaler = regenerate_artifacts(
                train_data_path, data_dir
            )
            
            # Save regenerated artifacts
            os.makedirs(artifacts_dir, exist_ok=True)
            joblib.dump(vocabulary, vocab_path)
            joblib.dump(user_categories, user_cat_path)
            joblib.dump(scaler, scaler_path)
            print(f"✓ Saved regenerated artifacts to {artifacts_dir}")
        else:
            raise FileNotFoundError(
                f"Artifacts not found and no training data path provided.\n"
                f"Please provide --train-path argument to regenerate artifacts from training data."
            )
    else:
        vocabulary = joblib.load(vocab_path)
        user_categories = joblib.load(user_cat_path)
        scaler = joblib.load(scaler_path)
    
    print(f"✓ Loaded vocabulary: {len(vocabulary)} tokens")
    print(f"✓ Loaded user categories: {len(user_categories.categories[0])} users")
    print(f"✓ Loaded statistical feature scaler")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        print(f"✓ Loaded configuration from: {config_path}")
    else:
        # Try to reconstruct config from checkpoint
        print(f"⚠️  Config file not found, reconstructing from checkpoint...")
        config = reconstruct_config_from_checkpoint(checkpoint_path)
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model - Get parameters from checkpoint hyperparameters first
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        vocab_size = hparams.get('vocab_size', len(vocabulary))
        num_users = hparams.get('num_users', len(user_categories.categories[0]) if hasattr(user_categories, 'categories') else len(set(user_categories)))
        stat_feature_dim = hparams.get('stat_feature_dim', len(scaler.mean_) if hasattr(scaler, 'mean_') else 19)
        print(f"✓ Using hyperparameters from checkpoint:")
        print(f"  • vocab_size: {vocab_size}")
        print(f"  • num_users: {num_users}")
        print(f"  • stat_feature_dim: {stat_feature_dim}")
    else:
        vocab_size = len(vocabulary)
        num_users = len(user_categories.categories[0]) if hasattr(user_categories, 'categories') else len(set(user_categories))
        stat_feature_dim = len(scaler.mean_) if hasattr(scaler, 'mean_') else 19
        print(f"⚠️  No hyperparameters in checkpoint, using inferred values:")
        print(f"  • vocab_size: {vocab_size}")
        print(f"  • num_users: {num_users}")
        print(f"  • stat_feature_dim: {stat_feature_dim}")
    
    # Detect which architecture based on state_dict keys
    state_dict = checkpoint['state_dict']
    has_complex_classifier = any('classifier_block1' in k for k in state_dict.keys())
    has_simple_classifier = any('classifier_layers' in k for k in state_dict.keys())
    
    if has_simple_classifier:
        print(f"✓ Detected SIMPLE classifier architecture")
        # Use simple architecture
        model = AttentionLSTMClassifierSimple(
            vocab_size=vocab_size,
            num_users=num_users,
            stat_feature_dim=stat_feature_dim,
            browser_dim=4,
            embedding_dim=config.model.embedding_dim,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            bidirectional=config.model.bidirectional,
            attention_dim=config.model.attention_dim,
            fusion_hidden_size=config.model.fusion_hidden_size
        )
    elif has_complex_classifier:
        print(f"✓ Detected classifier architecture")
        # Use complex architecture
        model = AttentionLSTMClassifier(
            config=config,
            vocab_size=vocab_size,
            num_users=num_users,
            stat_feature_dim=stat_feature_dim
        )
    else:
        raise ValueError("Unable to detect classifier architecture from checkpoint")
    
    # Load weights (strict=False to skip criterion and other training-only parameters)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys (usually OK): {missing_keys[:5]}...")
    if unexpected_keys:
        # Filter out expected training-only keys
        training_only_keys = [k for k in unexpected_keys if 'criterion' in k or 'optimizer' in k]
        other_unexpected = [k for k in unexpected_keys if k not in training_only_keys]
        if other_unexpected:
            print(f"⚠️  Unexpected keys: {other_unexpected[:5]}...")
    
    model.eval()
    model.to(device)
    
    print(f"✓ Model loaded successfully")
    print(f"  • Device: {device}")
    print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Statistical feature dim: {stat_feature_dim}")
    
    return model, vocabulary, user_categories, scaler, config


def reconstruct_config_from_checkpoint(checkpoint_path):
    """
    Reconstruct configuration from checkpoint when config.yaml is missing.
    
    Args:
        checkpoint_path (str): Path to checkpoint
        
    Returns:
        OmegaConf: Reconstructed configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Extract architecture parameters from state_dict
    embedding_weight = state_dict.get('embedding.weight')
    vocab_size = embedding_weight.shape[0] if embedding_weight is not None else 4000
    embedding_dim = embedding_weight.shape[1] if embedding_weight is not None else 64
    
    # Try to infer LSTM parameters
    lstm_keys = [k for k in state_dict.keys() if 'lstm' in k and 'weight_ih_l0' in k]
    if lstm_keys:
        hidden_size = state_dict[lstm_keys[0]].shape[0] // 4  # LSTM has 4 gates
    else:
        hidden_size = 128  # Default
    
    # Try to infer number of LSTM layers
    num_layers = 0
    for key in state_dict.keys():
        if 'lstm.weight_ih_l' in key:
            layer_num = int(key.split('lstm.weight_ih_l')[1].split('.')[0])
            num_layers = max(num_layers, layer_num + 1)
    
    if num_layers == 0:
        num_layers = 1  # Default
    
    # Infer bidirectional
    bidirectional = any('lstm.weight_hh_l0_reverse' in k for k in state_dict.keys())
    
    # Try to infer attention dimension
    attention_keys = [k for k in state_dict.keys() if 'attention' in k and 'weight' in k]
    attention_dim = 128  # Default
    if attention_keys:
        # Get dimension from first attention layer
        for key in attention_keys:
            if 'weight' in key and len(state_dict[key].shape) == 2:
                attention_dim = state_dict[key].shape[0]
                break
    
    # Try to infer fusion hidden size
    fusion_keys = [k for k in state_dict.keys() if 'fusion.0.weight' in k]
    fusion_hidden_size = 512  # Default
    if fusion_keys:
        fusion_hidden_size = state_dict[fusion_keys[0]].shape[0]
    
    # Create minimal config
    config_dict = {
        'model': {
            'embedding_dim': int(embedding_dim),
            'hidden_size': int(hidden_size),
            'num_layers': int(num_layers),
            'dropout': 0.2,
            'bidirectional': bool(bidirectional),
            'attention_dim': int(attention_dim),
            'fusion_hidden_size': int(fusion_hidden_size)
        },
        'data': {
            'max_sequence_length': 2000,
            'batch_size': 64
        },
        'training': {
            'use_focal_loss': False,
            'use_class_weights': True
        },
        'optimizer': {
            'name': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0001
        },
        'scheduler': {
            'name': 'reduce_on_plateau',
            'reduce_on_plateau': {
                'monitor': 'val_f1',
                'factor': 0.5,
                'patience': 4,
                'min_lr': 0.00001,
                'mode': 'max'
            }
        }
    }
    
    config = OmegaConf.create(config_dict)
    print(f"✓ Reconstructed configuration from checkpoint")
    print(f"  • Embedding dim: {embedding_dim}")
    print(f"  • Hidden size: {hidden_size}")
    print(f"  • Num layers: {num_layers}")
    print(f"  • Bidirectional: {bidirectional}")
    print(f"  • Attention dim: {attention_dim}")
    print(f"  • Fusion hidden size: {fusion_hidden_size}")
    
    return config


def prepare_test_data(test_df, vocabulary, scaler, max_length):
    """
    Prepare test data for prediction.
    
    Args:
        test_df (pd.DataFrame): Test dataframe
        vocabulary (dict): Action vocabulary
        scaler: Statistical feature scaler
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (sequences, browser_features, stat_features)
    """
    print(f"\n🔧 Preparing test data...")
    
    # Tokenize and prepare sequences (needed for statistical features)
    action_sequences, _ = tokenize_actions(test_df, is_train=False, vocabulary=vocabulary)
    
    # Prepare RNN sequences with browser features
    sequences, browser_features, _, _ = prepare_rnn_sequences(
        test_df, vocabulary, max_length=max_length, is_train=False
    )
    
    # Extract statistical features
    stat_features = extract_statistical_features(test_df, action_sequences=action_sequences)
    
    # Normalize statistical features using training scaler
    stat_features_normalized = scaler.transform(stat_features)
    
    print(f"✓ Test sequences shape: {sequences.shape}")
    print(f"✓ Browser features shape: {browser_features.shape}")
    print(f"✓ Statistical features shape: {stat_features_normalized.shape}")
    
    return sequences, browser_features, stat_features_normalized


def generate_predictions(model, sequences, stat_features, browser_features, batch_size=32):
    """
    Generate predictions for test data.
    
    Args:
        model: Trained AttentionLSTMClassifier
        sequences (np.ndarray): Test sequences
        stat_features (np.ndarray): Statistical features
        browser_features (np.ndarray): Browser features
        batch_size (int): Batch size for inference
        
    Returns:
        np.ndarray: Predicted user codes
    """
    print(f"\n🔮 Generating predictions...")
    
    device = next(model.parameters()).device
    num_samples = len(sequences)
    all_predictions = []
    
    # Process in batches
    for i in tqdm(range(0, num_samples, batch_size), desc="Predicting"):
        batch_end = min(i + batch_size, num_samples)
        
        # Get batch
        batch_sequences = torch.LongTensor(sequences[i:batch_end]).to(device)
        batch_stat = torch.FloatTensor(stat_features[i:batch_end]).to(device)
        batch_browser = torch.FloatTensor(browser_features[i:batch_end]).to(device)
        
        # Predict
        with torch.no_grad():
            batch_predictions = model.predict(batch_sequences, batch_stat, batch_browser)
        
        all_predictions.append(batch_predictions)
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    print(f"✓ Generated {len(predictions)} predictions")
    
    return predictions


def save_submission(predictions, user_categories, output_path="submission.csv"):
    """
    Save predictions to submission file.
    
    Args:
        predictions (np.ndarray): Predicted user codes
        user_categories: User category mapping (pandas Categorical)
        output_path (str): Output file path
    """
    print(f"\n💾 Saving submission...")
    
    # Map codes back to user IDs
    try:
        # Try standard categorical indexing
        if hasattr(user_categories, 'categories'):
            user_ids = user_categories.categories[predictions]
        else:
            # Fallback: assume it's already the mapping array
            user_ids = user_categories[predictions]
    except (TypeError, IndexError) as e:
        print(f"⚠️  Standard categorical indexing failed: {e}")
        print(f"   Trying alternative approach...")
        
        # Alternative: just use predictions as-is and hope they make sense
        # or create a mapping from codes
        if hasattr(user_categories, 'categories') and len(user_categories.categories) > 0:
            cats = user_categories.categories
            if hasattr(cats, '__getitem__'):
                # It's indexable - likely a pandas Index or array
                user_ids = np.array([cats[p] if p < len(cats) else f"user_{p}" for p in predictions])
            else:
                # Fallback to predictions as-is
                user_ids = predictions
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
    """Main inference function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Attention-LSTM User Identification - Inference')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--output', '-o', type=str, default='submission.csv',
                       help='Output submission file path (default: submission.csv)')
    parser.add_argument('--test-path', type=str, default='test',
                       help='Test data path (default: test)')
    parser.add_argument('--train-path', type=str, default='train',
                       help='Training data path for regenerating artifacts (default: train)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference (default: 64)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Attention-LSTM User Identification - Inference")
    print("=" * 80)
    
    # Load model and artifacts (will regenerate if missing)
    model, vocabulary, user_categories, scaler, config = load_model_and_artifacts(
        args.checkpoint, 
        train_data_path=args.train_path,
        data_dir=args.data_dir
    )
    
    # Load test data
    print(f"\n📂 Loading test data...")
    print(f"   Test path: {args.test_path}")
    print(f"   Data directory: {args.data_dir}")
    
    test_df = load_data(args.test_path, data_dir=args.data_dir, size='full')
    print(f"✓ Loaded {len(test_df)} test samples")
    
    # Clean test data
    test_df_clean = clean_data(test_df, is_train=False)
    
    # Prepare test data (sequences + browser features + statistical features)
    max_length = config.data.max_sequence_length
    sequences, browser_features, stat_features = prepare_test_data(
        test_df_clean, vocabulary, scaler, max_length
    )
    
    # Generate predictions
    predictions = generate_predictions(model, sequences, stat_features, browser_features, 
                                       batch_size=args.batch_size)
    
    # Save submission
    save_submission(predictions, user_categories, output_path=args.output)
    
    print("\n" + "=" * 80)
    print("Inference completed! 🎉")
    print("=" * 80)
    print(f"\n✓ Submission file ready: {args.output}")


if __name__ == "__main__":
    main()

