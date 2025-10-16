"""
Training script for RNN-based user identification using PyTorch Lightning and Hydra.

This script uses:
- Hydra for configuration management
- PyTorch Lightning for training
- Model checkpointing (best.ckpt and last.ckpt)
- Learning rate scheduling
- JSON metrics logging
"""

import os
import json
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from collections import Counter

from utils import load_data, clean_data, tokenize_actions, prepare_rnn_sequences
from rnn_lightning import RNNUserClassifierLightning


def create_data_loaders_lightning(sequences, browser_features, targets, config):
    """
    Create PyTorch DataLoaders for Lightning training.
    
    Args:
        sequences (np.ndarray): Action sequences
        browser_features (np.ndarray): Browser features
        targets (np.ndarray): Target user IDs
        config: Hydra configuration
        
    Returns:
        tuple: (train_loader, val_loader, train_indices, val_indices)
    """
    # Check if stratified splitting is possible
    class_counts = Counter(targets)
    min_class_count = min(class_counts.values())
    use_stratify = min_class_count >= 2
    
    if not use_stratify:
        print(f"⚠️  Warning: {sum(1 for c in class_counts.values() if c < 2)} classes have < 2 samples. Using non-stratified split.")
    
    # Split data
    indices = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=(1 - config.data.train_split), 
        random_state=config.seed,
        stratify=targets if use_stratify else None
    )
    
    # Create training dataset
    train_sequences = torch.LongTensor(sequences[train_idx])
    train_browser = torch.FloatTensor(browser_features[train_idx])
    train_targets = torch.LongTensor(targets[train_idx])
    train_dataset = TensorDataset(train_sequences, train_browser, train_targets)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True,
        num_workers=config.data.num_workers,
        persistent_workers=False if config.data.num_workers == 0 else True
    )
    
    # Create validation dataset
    val_sequences = torch.LongTensor(sequences[val_idx])
    val_browser = torch.FloatTensor(browser_features[val_idx])
    val_targets = torch.LongTensor(targets[val_idx])
    val_dataset = TensorDataset(val_sequences, val_browser, val_targets)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False,
        num_workers=config.data.num_workers,
        persistent_workers=False if config.data.num_workers == 0 else True
    )
    
    return train_loader, val_loader, train_idx, val_idx


def save_results_to_json(config, model, train_metrics, val_metrics, vocab_size, 
                         num_users, checkpoint_path, output_dir):
    """
    Save training results and metrics to JSON file.
    
    Args:
        config: Hydra configuration
        model: Trained Lightning model
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        vocab_size: Size of vocabulary
        num_users: Number of users
        checkpoint_path: Path to best checkpoint
        output_dir: Output directory for results
    """
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': 'rnn_user_classifier_lightning',
        'config': OmegaConf.to_container(config, resolve=True),
        'data': {
            'vocab_size': vocab_size,
            'num_users': num_users,
            'max_sequence_length': config.data.max_sequence_length,
            'data_size': config.data.size
        },
        'architecture': {
            'embedding_dim': config.model.embedding_dim,
            'hidden_size': config.model.hidden_size,
            'num_layers': config.model.num_layers,
            'dropout': config.model.dropout,
            'bidirectional': config.model.bidirectional
        },
        'training': {
            'optimizer': config.optimizer.name,
            'learning_rate': config.optimizer.lr,
            'weight_decay': config.optimizer.weight_decay,
            'scheduler': config.scheduler.name,
            'max_epochs': config.training.max_epochs,
            'batch_size': config.data.batch_size
        },
        'metrics': {
            'train': train_metrics,
            'validation': val_metrics
        },
        'checkpoint': {
            'best_model_path': checkpoint_path
        }
    }
    
    # Save to JSON
    os.makedirs(config.paths.metrics_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(config.paths.metrics_dir, f'rnn_lightning_{timestamp}_metrics.json')
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {json_path}")
    
    return json_path


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """
    Main training function with Hydra configuration.
    
    Args:
        config: Hydra configuration object
    """
    print("=" * 80)
    print("RNN User Identification - PyTorch Lightning + Hydra")
    print("=" * 80)
    
    # Print configuration
    print("\n📋 Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Set random seeds
    pl.seed_everything(config.seed)
    
    # ========================================================================
    # 1. LOAD AND PREPARE DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("[1/6] Loading and preparing data...")
    print("=" * 80)
    
    df_train = load_data(config.data.train_path, data_dir=config.data.data_dir, size=config.data.size)
    print(f"✓ Loaded {len(df_train)} training samples")
    
    df_train_clean = clean_data(df_train, is_train=True)
    num_users = df_train_clean['user_id'].nunique()
    print(f"✓ Number of unique users: {num_users}")
    
    # ========================================================================
    # 2. BUILD VOCABULARY AND TOKENIZE
    # ========================================================================
    print("\n" + "=" * 80)
    print("[2/6] Building vocabulary and tokenizing sequences...")
    print("=" * 80)
    
    sequences, vocabulary = tokenize_actions(df_train_clean, is_train=True)
    vocab_size = len(vocabulary)
    print(f"✓ Vocabulary size: {vocab_size} unique action signatures")
    print(f"✓ Average sequence length: {np.mean([len(s) for s in sequences]):.1f} actions")
    print(f"✓ Max sequence length: {max([len(s) for s in sequences])} actions")
    
    # ========================================================================
    # 3. PREPARE RNN SEQUENCES
    # ========================================================================
    print("\n" + "=" * 80)
    print("[3/6] Preparing RNN sequences...")
    print("=" * 80)
    
    X_sequences, X_browser, y, user_categories = prepare_rnn_sequences(
        df_train_clean, vocabulary, max_length=config.data.max_sequence_length, is_train=True
    )
    print(f"✓ Sequences shape: {X_sequences.shape}")
    print(f"✓ Browser features shape: {X_browser.shape}")
    print(f"✓ Targets shape: {y.shape}")
    
    # ========================================================================
    # 4. CREATE DATA LOADERS
    # ========================================================================
    print("\n" + "=" * 80)
    print("[4/6] Creating data loaders...")
    print("=" * 80)
    
    train_loader, val_loader, train_idx, val_idx = create_data_loaders_lightning(
        X_sequences, X_browser, y, config
    )
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print(f"✓ Training samples: {len(train_idx)}")
    print(f"✓ Validation samples: {len(val_idx)}")
    
    # ========================================================================
    # 5. INITIALIZE MODEL AND CALLBACKS
    # ========================================================================
    print("\n" + "=" * 80)
    print("[5/6] Initializing model and training components...")
    print("=" * 80)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"rnn_lightning_{timestamp}"
    output_dir = os.path.join(config.paths.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Initialize model
    model = RNNUserClassifierLightning(config, vocab_size, num_users)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total trainable parameters: {total_params:,}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=config.training.checkpoint.filename,
        monitor=config.training.checkpoint.monitor,
        mode=config.training.checkpoint.mode,
        save_top_k=config.training.checkpoint.save_top_k,
        save_last=config.training.checkpoint.save_last,
        verbose=True
    )
    
    early_stopping_callback = EarlyStopping(
        monitor=config.training.early_stopping.monitor,
        patience=config.training.early_stopping.patience,
        mode=config.training.early_stopping.mode,
        min_delta=config.training.early_stopping.min_delta,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config.paths.logs_dir,
        name=model_name
    )
    
    print(f"✓ Callbacks configured:")
    print(f"  - ModelCheckpoint (monitor: {config.training.checkpoint.monitor})")
    print(f"  - EarlyStopping (patience: {config.training.early_stopping.patience})")
    print(f"  - LearningRateMonitor")
    print(f"  - TensorBoardLogger")
    
    # ========================================================================
    # 6. TRAIN MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("[6/6] Training model...")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  • Max epochs: {config.training.max_epochs}")
    print(f"  • Batch size: {config.data.batch_size}")
    print(f"  • Learning rate: {config.optimizer.lr}")
    print(f"  • Scheduler: {config.scheduler.name}")
    print(f"  • Optimizer: {config.optimizer.name}")
    print(f"  • Device: {config.training.accelerator}")
    print()
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # ========================================================================
    # POST-TRAINING: SAVE RESULTS AND ARTIFACTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Get best metrics
    best_val_acc = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0
    print(f"\n✓ Best validation accuracy: {best_val_acc:.4f}")
    print(f"✓ Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"✓ Last model checkpoint: {checkpoint_callback.last_model_path}")
    
    # Load best model for final evaluation
    best_model = RNNUserClassifierLightning.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        config=config,
        vocab_size=vocab_size,
        num_users=num_users
    )
    best_model.eval()
    
    # Get predictions on train and validation sets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.to(device)
    
    with torch.no_grad():
        # Training predictions
        train_sequences_tensor = torch.LongTensor(X_sequences[train_idx]).to(device)
        train_browser_tensor = torch.FloatTensor(X_browser[train_idx]).to(device)
        train_predictions = best_model.predict(train_sequences_tensor, train_browser_tensor)
        
        # Validation predictions
        val_sequences_tensor = torch.LongTensor(X_sequences[val_idx]).to(device)
        val_browser_tensor = torch.FloatTensor(X_browser[val_idx]).to(device)
        val_predictions = best_model.predict(val_sequences_tensor, val_browser_tensor)
        val_predictions_proba = best_model.predict_proba(val_sequences_tensor, val_browser_tensor)
    
    # Calculate final metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    train_metrics = {
        'accuracy': float(accuracy_score(y[train_idx], train_predictions)),
        'f1_weighted': float(f1_score(y[train_idx], train_predictions, average='weighted')),
        'f1_macro': float(f1_score(y[train_idx], train_predictions, average='macro')),
        'precision': float(precision_score(y[train_idx], train_predictions, average='weighted', zero_division=0)),
        'recall': float(recall_score(y[train_idx], train_predictions, average='weighted'))
    }
    
    val_metrics = {
        'accuracy': float(accuracy_score(y[val_idx], val_predictions)),
        'f1_weighted': float(f1_score(y[val_idx], val_predictions, average='weighted')),
        'f1_macro': float(f1_score(y[val_idx], val_predictions, average='macro')),
        'precision': float(precision_score(y[val_idx], val_predictions, average='weighted', zero_division=0)),
        'recall': float(recall_score(y[val_idx], val_predictions, average='weighted'))
    }
    
    print(f"\n📊 Final Metrics:")
    print(f"\nTraining:")
    for metric, value in train_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    print(f"\nValidation:")
    for metric, value in val_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    # Save results to JSON
    json_path = save_results_to_json(
        config, best_model, train_metrics, val_metrics, 
        vocab_size, num_users, checkpoint_callback.best_model_path, output_dir
    )
    
    # Save vocabulary and artifacts
    artifacts_path = os.path.join(output_dir, 'artifacts')
    os.makedirs(artifacts_path, exist_ok=True)
    
    joblib.dump(vocabulary, os.path.join(artifacts_path, 'vocabulary.joblib'))
    joblib.dump(user_categories, os.path.join(artifacts_path, 'user_categories.joblib'))
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config, f)
    
    print(f"\n✓ Vocabulary saved to: {artifacts_path}/vocabulary.joblib")
    print(f"✓ User categories saved to: {artifacts_path}/user_categories.joblib")
    print(f"✓ Configuration saved to: {output_dir}/config.yaml")
    
    print("\n" + "=" * 80)
    print("All done! 🎉")
    print("=" * 80)
    
    return train_metrics, val_metrics


if __name__ == "__main__":
    main()

