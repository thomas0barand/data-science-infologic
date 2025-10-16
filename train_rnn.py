"""
Training script for RNN-based user identification.

This script trains a Simple RNN model to predict users from action sequences.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
from utils import load_data, clean_data, tokenize_actions, prepare_rnn_sequences
from rnn_model import SimpleRNNUserClassifier, create_data_loaders, train_epoch, evaluate
from model_manager import ModelManager, save_model_results


def main():
    print("=" * 80)
    print("Copilote User Identification - Simple RNN Training")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize model manager
    manager = ModelManager()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load training data
    print("\n[1/8] Loading training data...")
    df_train = load_data("train", size="small")
    print(f"   ✓ Loaded {len(df_train)} training samples")
    print(f"   ✓ Data shape: {df_train.shape}")
    
    # Clean data
    print("\n[2/8] Cleaning data...")
    df_train_clean = clean_data(df_train, is_train=True)
    print(f"   ✓ Data cleaned")
    print(f"   ✓ Number of unique users: {df_train_clean['user_id'].nunique()}")
    
    # Build vocabulary
    print("\n[3/8] Building vocabulary from action sequences...")
    sequences, vocabulary = tokenize_actions(df_train_clean, is_train=True)
    vocab_size = len(vocabulary)
    print(f"   ✓ Vocabulary size: {vocab_size} unique actions")
    print(f"   ✓ Average sequence length: {np.mean([len(s) for s in sequences]):.1f} actions")
    print(f"   ✓ Max sequence length: {max([len(s) for s in sequences])} actions")
    
    # Prepare RNN sequences
    max_length = 5000
    print(f"\n[4/8] Preparing sequences (truncate to {max_length}, pad, encode browser)...")
    
    X_sequences, X_browser, y, user_categories = prepare_rnn_sequences(
        df_train_clean, vocabulary, max_length=max_length, is_train=True
    )
    print(f"   ✓ Sequences shape: {X_sequences.shape}")
    print(f"   ✓ Browser features shape: {X_browser.shape}")
    print(f"   ✓ Targets shape: {y.shape}")
    print(f"   ✓ Number of classes: {len(np.unique(y))}")
    
    # Create data loaders
    print("\n[5/8] Creating train/validation data loaders...")
    batch_size = 16
    train_loader, val_loader, train_idx, val_idx = create_data_loaders(
        X_sequences, X_browser, y, batch_size=batch_size, 
        train_split=0.8, random_state=42
    )
    print(f"   ✓ Training batches: {len(train_loader)}")
    print(f"   ✓ Validation batches: {len(val_loader)}")
    print(f"   ✓ Training samples: {len(train_idx)}")
    print(f"   ✓ Validation samples: {len(val_idx)}")
    
    # Initialize model
    print("\n[6/8] Initializing Simple RNN model...")
    num_users = len(np.unique(y))
    model = SimpleRNNUserClassifier(
        vocab_size=vocab_size,
        num_users=num_users,
        embedding_dim=128,
        hidden_size=256,
        browser_dim=4,
        dropout=0.3
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model initialized")
    print(f"   ✓ Total trainable parameters: {num_params:,}")
    print(f"   ✓ Embedding dim: 128, Hidden size: 256")
    
    # Setup training
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 25
    
    print(f"\n[7/8] Training model for {num_epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Optimizer: Adam")
    
    best_val_accuracy = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Prepare unique folder name for saving checkpoints
    model_name = manager.generate_model_name("simple_rnn")
    model_dir = os.path.join("models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    best_val_accuracy = 0
    best_epoch = 0
    best_ckpt_path = os.path.join(model_dir, "best.ckpt")
    last_ckpt_path = os.path.join(model_dir, "last.ckpt")

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_acc, val_predictions, val_targets = evaluate(model, val_loader, criterion, device)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"   Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Track best model and save checkpoint if improved
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'train_loss': train_loss,
                'history': history,
                'hyperparameters': {
                    'embedding_dim': 128,
                    'hidden_size': 256,
                    'dropout': 0.3,
                    'browser_dim': 4,
                    'batch_size': batch_size,
                },
            }, best_ckpt_path)

        # Always save the last checkpoint at each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'history': history,
            'hyperparameters': {
                'embedding_dim': 128,
                'hidden_size': 256,
                'dropout': 0.3,
                'browser_dim': 4,
                'batch_size': batch_size,
            },
        }, last_ckpt_path)
    
    print(f"\n   ✓ Training completed!")
    print(f"   ✓ Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
    
    # Final evaluation
    print("\n[8/8] Final evaluation on full train and validation sets...")
    
    # Get predictions for full training set
    model.eval()
    with torch.no_grad():
        train_sequences_tensor = torch.LongTensor(X_sequences[train_idx]).to(device)
        train_browser_tensor = torch.FloatTensor(X_browser[train_idx]).to(device)
        train_predictions = model.predict(train_sequences_tensor, train_browser_tensor)
    
    # Get predictions for validation set
    with torch.no_grad():
        val_sequences_tensor = torch.LongTensor(X_sequences[val_idx]).to(device)
        val_browser_tensor = torch.FloatTensor(X_browser[val_idx]).to(device)
        val_predictions = model.predict(val_sequences_tensor, val_browser_tensor)
        val_predictions_proba = model.predict_proba(val_sequences_tensor, val_browser_tensor)
    
    # Calculate comprehensive metrics
    train_metrics = manager.calculate_comprehensive_metrics(y[train_idx], train_predictions)
    val_metrics = manager.calculate_comprehensive_metrics(y[val_idx], val_predictions, val_predictions_proba)
    
    print(f"\n   Training Metrics:")
    print(f"   • Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"   • F1-Score (weighted): {train_metrics['f1_weighted']:.4f}")
    print(f"   • F1-Score (macro): {train_metrics['f1_macro']:.4f}")
    
    print(f"\n   Validation Metrics:")
    print(f"   • Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"   • F1-Score (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"   • F1-Score (macro): {val_metrics['f1_macro']:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    
    # Save model and artifacts
    print("\n[9/9] Saving model, vocabulary, and metrics...")
    
    # Create model directory
    model_name = manager.generate_model_name("simple_rnn")
    model_dir = os.path.join(manager.models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save PyTorch model
    model_path = os.path.join(model_dir, f"{model_name}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'num_users': num_users,
        'embedding_dim': 128,
        'hidden_size': 256,
        'browser_dim': 4,
        'max_length': max_length
    }, model_path)
    
    # Save vocabulary
    vocab_path = os.path.join(model_dir, f"{model_name}_vocabulary.joblib")
    joblib.dump(vocabulary, vocab_path)
    
    # Save user categories
    categories_path = os.path.join(model_dir, f"{model_name}_categories.joblib")
    joblib.dump(user_categories, categories_path)
    
    # Save training history
    history_path = os.path.join(model_dir, f"{model_name}_history.joblib")
    joblib.dump(history, history_path)
    
    # Prepare metadata
    additional_metadata = {
        "vocab_size": vocab_size,
        "num_users": num_users,
        "embedding_dim": 128,
        "hidden_size": 256,
        "browser_dim": 4,
        "max_length": max_length,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": 0.001,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "num_parameters": num_params,
        "model_architecture": "Simple RNN",
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss"
    }
    
    # Save metadata
    import json
    from datetime import datetime
    metadata = {
        "model_name": model_name,
        "model_type": "simple_rnn",
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "vocabulary_path": vocab_path,
        "categories_path": categories_path,
        "history_path": history_path,
        "additional_metadata": additional_metadata
    }
    
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save metrics
    metrics_path = manager.save_metrics(
        model_name=model_name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        additional_info=additional_metadata
    )
    
    print(f"   ✓ Model saved to: {model_path}")
    print(f"   ✓ Vocabulary saved to: {vocab_path}")
    print(f"   ✓ Metrics saved to: {metrics_path}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Model Name: {model_name}")
    print(f"Model Type: Simple RNN")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Number of Users: {num_users}")
    print(f"Sequence Length: {max_length}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Training F1 (weighted): {train_metrics['f1_weighted']:.4f}")
    print(f"Validation F1 (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"Best Epoch: {best_epoch}/{num_epochs}")
    print("=" * 80)
    
    return model, vocabulary, user_categories


if __name__ == "__main__":
    main()

