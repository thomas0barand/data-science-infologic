"""
Inference script for Transformer User Identification

This script generates predictions on test data using a trained Transformer model.

Usage:
    python inference_transformer.py --checkpoint results/transformer_models/transformer_20251020_120000/best.ckpt
    python inference_transformer.py --checkpoint results/transformer_models/transformer_20251020_120000/best.ckpt --output my_submission.csv
    python inference_transformer.py -c results/transformer_models/transformer_20251020_120000/best.ckpt
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
    prepare_rnn_sequences
)
from transformer_lightning import TransformerUserClassifier


def load_model_and_artifacts(checkpoint_path):
    """
    Load trained model and required artifacts.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        
    Returns:
        tuple: (model, vocabulary, user_categories, config)
    """
    print(f"\n📦 Loading model and artifacts from checkpoint...")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Get directory containing checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    artifacts_dir = os.path.join(checkpoint_dir, 'artifacts')
    
    # Load vocabulary and user categories
    vocab_path = os.path.join(artifacts_dir, 'vocabulary.joblib')
    user_cat_path = os.path.join(artifacts_dir, 'user_categories.joblib')
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at: {vocab_path}")
    if not os.path.exists(user_cat_path):
        raise FileNotFoundError(f"User categories not found at: {user_cat_path}")
    
    vocabulary = joblib.load(vocab_path)
    user_categories = joblib.load(user_cat_path)
    
    print(f"✓ Loaded vocabulary: {len(vocabulary)} tokens")
    print(f"✓ Loaded user categories: {len(user_categories.categories[0])} users")
    
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
    
    # Initialize model
    vocab_size = len(vocabulary)
    num_users = len(user_categories.categories[0])
    
    model = TransformerUserClassifier(
        config=config,
        vocab_size=vocab_size,
        num_users=num_users
    )
    
    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    
    print(f"✓ Model loaded successfully")
    print(f"  • Device: {device}")
    print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, vocabulary, user_categories, config


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
    d_model = embedding_weight.shape[1] if embedding_weight is not None else 256
    
    # Try to infer num_layers from transformer encoder
    num_layers = 0
    for key in state_dict.keys():
        if 'transformer_encoder.layers.' in key:
            layer_num = int(key.split('transformer_encoder.layers.')[1].split('.')[0])
            num_layers = max(num_layers, layer_num + 1)
    
    if num_layers == 0:
        num_layers = 4  # Default
    
    # Create minimal config
    config_dict = {
        'model': {
            'd_model': int(d_model),
            'nhead': 8,  # Default
            'num_layers': int(num_layers),
            'dim_feedforward': 1024,  # Default
            'dropout': 0.3,
            'local_attention_window': 128,
            'pooling': 'cls'
        },
        'data': {
            'max_sequence_length': 1000,
            'batch_size': 32
        },
        'training': {
            'use_focal_loss': False
        },
        'optimizer': {
            'name': 'adamw',
            'lr': 0.0001,
            'weight_decay': 0.01
        },
        'scheduler': {
            'name': 'cosine',
            'cosine': {
                'T_max': 30,
                'eta_min': 0.000001
            }
        }
    }
    
    config = OmegaConf.create(config_dict)
    print(f"✓ Reconstructed configuration from checkpoint")
    
    return config


def prepare_test_data(test_df, vocabulary, max_length):
    """
    Prepare test data for prediction.
    
    Args:
        test_df (pd.DataFrame): Test dataframe
        vocabulary (dict): Action vocabulary
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (sequences, browser_features)
    """
    print(f"\n🔧 Preparing test data...")
    
    # Tokenize and prepare sequences
    sequences, browser_features, _, _ = prepare_rnn_sequences(
        test_df, vocabulary, max_length=max_length, is_train=False
    )
    
    print(f"✓ Test sequences shape: {sequences.shape}")
    print(f"✓ Browser features shape: {browser_features.shape}")
    
    return sequences, browser_features


def generate_predictions(model, sequences, browser_features, batch_size=32):
    """
    Generate predictions for test data.
    
    Args:
        model: Trained TransformerUserClassifier
        sequences (np.ndarray): Test sequences
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
        batch_browser = torch.FloatTensor(browser_features[i:batch_end]).to(device)
        
        # Predict
        with torch.no_grad():
            batch_predictions = model.predict(batch_sequences, batch_browser)
        
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
        user_categories: User category mapping
        output_path (str): Output file path
    """
    print(f"\n💾 Saving submission...")
    
    # Map codes back to user IDs
    user_ids = user_categories.categories[0][predictions]
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'user_id': user_ids
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"✓ Submission saved to: {output_path}")
    print(f"  • Number of predictions: {len(submission_df)}")
    print(f"  • Unique users predicted: {submission_df['user_id'].nunique()}")
    
    # Show distribution
    value_counts = submission_df['user_id'].value_counts()
    print(f"  • Most frequent user: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
    print(f"  • Distribution: min={value_counts.min()}, max={value_counts.max()}, "
          f"median={value_counts.median():.1f}")


def main():
    """Main inference function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Transformer User Identification - Inference')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--output', '-o', type=str, default='submission.csv',
                       help='Output submission file path (default: submission.csv)')
    parser.add_argument('--test-path', type=str, default='test',
                       help='Test data path (default: test)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Transformer User Identification - Inference")
    print("=" * 80)
    
    # Load model and artifacts
    model, vocabulary, user_categories, config = load_model_and_artifacts(args.checkpoint)
    
    # Load test data
    print(f"\n📂 Loading test data...")
    print(f"   Test path: {args.test_path}")
    print(f"   Data directory: {args.data_dir}")
    
    test_df = load_data(args.test_path, data_dir=args.data_dir, size='full')
    print(f"✓ Loaded {len(test_df)} test samples")
    
    # Clean test data
    test_df_clean = clean_data(test_df, is_train=False)
    
    # Prepare test data (sequences + browser features only)
    max_length = config.data.max_sequence_length
    sequences, browser_features = prepare_test_data(test_df_clean, vocabulary, max_length)
    
    # Generate predictions
    predictions = generate_predictions(model, sequences, browser_features, 
                                       batch_size=args.batch_size)
    
    # Save submission
    save_submission(predictions, user_categories, output_path=args.output)
    
    print("\n" + "=" * 80)
    print("Inference completed! 🎉")
    print("=" * 80)
    print(f"\n✓ Submission file ready: {args.output}")


if __name__ == "__main__":
    main()

