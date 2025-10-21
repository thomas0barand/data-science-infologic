"""
Training script for Attention-LSTM User Identification

Enhanced training pipeline with:
- Attention-based LSTM architecture
- Statistical feature fusion
- Class imbalance handling (focal loss + class weights)
- Advanced metrics tracking
- Automatic checkpoint resume (if training is interrupted)

Usage:
    # Train with specific config
    python train_attention_lightning.py -c config_attention_large
    python train_attention_lightning.py --config-name config_attention_deep
    
    # Resume training automatically (if interrupted)
    python train_attention_lightning.py -c config_attention_large
    
    # Force fresh training (ignore existing checkpoints)
    python train_attention_lightning.py -c config_attention_large --fresh
"""

import os
import sys
import json
from datetime import datetime
import argparse
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
from sklearn.preprocessing import StandardScaler
from collections import Counter

from utils import (
    load_data, clean_data, tokenize_actions, 
    prepare_rnn_sequences, extract_statistical_features
)
from rnn_attention_lightning import AttentionLSTMClassifier
from cache_utils import (
    get_cache_paths, check_cache_complete, print_cache_info,
    load_stage1_cache, save_stage1_cache,
    load_stage2_cache, save_stage2_cache,
    load_stage3_cache, save_stage3_cache,
    load_stage4_cache, save_stage4_cache,
    save_metadata, load_metadata
)


def compute_class_weights(targets, num_classes, device='cpu'):
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        targets (np.ndarray): Target labels
        num_classes (int): Number of classes
        device (str): Device to put weights on
        
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = np.bincount(targets, minlength=num_classes)
    
    # Compute inverse frequency weights
    # Add 1 to avoid division by zero
    total_samples = len(targets)
    weights = total_samples / (num_classes * (class_counts + 1))
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return torch.FloatTensor(weights).to(device)


def create_data_loaders_with_features(sequences, browser_features, stat_features, 
                                     targets, config):
    """
    Create PyTorch DataLoaders with all features.
    
    Args:
        sequences (np.ndarray): Action sequences
        browser_features (np.ndarray): Browser features
        stat_features (np.ndarray): Statistical features
        targets (np.ndarray): Target user IDs
        config: Hydra configuration
        
    Returns:
        tuple: (train_loader, val_loader, train_indices, val_indices, scaler)
    """
    # Check if stratified splitting is possible
    class_counts = Counter(targets)
    min_class_count = min(class_counts.values())
    use_stratify = min_class_count >= 2
    
    if not use_stratify:
        print(f"⚠️  Warning: {sum(1 for c in class_counts.values() if c < 2)} classes have < 2 samples.")
        print("   Using non-stratified split.")
    
    # Split data
    indices = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=(1 - config.data.train_split),
        random_state=config.seed,
        stratify=targets if use_stratify else None
    )
    
    # Normalize statistical features using training set statistics
    scaler = StandardScaler()
    stat_features_train = scaler.fit_transform(stat_features[train_idx])
    stat_features_val = scaler.transform(stat_features[val_idx])
    
    print(f"✓ Statistical features normalized (mean=0, std=1)")
    print(f"  • Feature dimension: {stat_features.shape[1]}")
    
    # Create training dataset
    train_sequences = torch.LongTensor(sequences[train_idx])
    train_stat = torch.FloatTensor(stat_features_train)
    train_browser = torch.FloatTensor(browser_features[train_idx])
    train_targets = torch.LongTensor(targets[train_idx])
    train_dataset = TensorDataset(train_sequences, train_stat, train_browser, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        persistent_workers=False if config.data.num_workers == 0 else True
    )
    
    # Create validation dataset
    val_sequences = torch.LongTensor(sequences[val_idx])
    val_stat = torch.FloatTensor(stat_features_val)
    val_browser = torch.FloatTensor(browser_features[val_idx])
    val_targets = torch.LongTensor(targets[val_idx])
    val_dataset = TensorDataset(val_sequences, val_stat, val_browser, val_targets)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        persistent_workers=False if config.data.num_workers == 0 else True
    )
    
    return train_loader, val_loader, train_idx, val_idx, scaler


def save_results_to_json(config, model, train_metrics, val_metrics, vocab_size,
                         num_users, checkpoint_path, output_dir, model_name=None):
    """Save training results and metrics to JSON file."""
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name if model_name else 'attention_lstm_user_classifier',
        'config': OmegaConf.to_container(config, resolve=True),
        'data': {
            'vocab_size': vocab_size,
            'num_users': num_users,
            'max_sequence_length': config.data.max_sequence_length,
            'data_size': config.data.size
        },
        'architecture': {
            'model_type': 'Attention-LSTM',
            'embedding_dim': config.model.embedding_dim,
            'hidden_size': config.model.hidden_size,
            'num_layers': config.model.num_layers,
            'dropout': config.model.dropout,
            'bidirectional': config.model.bidirectional,
            'attention_dim': config.model.attention_dim,
            'fusion_hidden_size': config.model.fusion_hidden_size,
            'use_focal_loss': config.training.use_focal_loss
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
    
    # Save to JSON with model name
    os.makedirs(config.paths.metrics_dir, exist_ok=True)
    
    # Use model_name if provided, otherwise use timestamp
    if model_name and model_name != 'attention_lstm_user_classifier':
        json_filename = f'{model_name}_metrics.json'
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = f'attention_lstm_{timestamp}_metrics.json'
    
    json_path = os.path.join(config.paths.metrics_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {json_path}")
    
    return json_path


def parse_custom_args():
    """
    Parse custom command-line arguments before Hydra processes them.
    Converts -c config_name to Hydra's --config-name format.
    Returns the config name and force_fresh flag.
    """
    config_name = None
    force_fresh = False
    
    # Check if -c argument is present
    if '-c' in sys.argv:
        idx = sys.argv.index('-c')
        if idx + 1 < len(sys.argv):
            config_name = sys.argv[idx + 1]
            # Remove -c and its value from sys.argv
            sys.argv.pop(idx)  # Remove -c
            sys.argv.pop(idx)  # Remove config_name
            # Add Hydra format
            sys.argv.append(f'--config-name={config_name}')
    
    # Check if --config-name is present
    elif any(arg.startswith('--config-name=') for arg in sys.argv):
        for arg in sys.argv:
            if arg.startswith('--config-name='):
                config_name = arg.split('=')[1]
                break
    
    # Check for --fresh flag to force new training
    if '--fresh' in sys.argv:
        sys.argv.remove('--fresh')
        force_fresh = True
    
    # If no config name found, check Hydra's default
    if config_name is None:
        # Will use default from decorator
        config_name = "config_attention"
    
    return config_name, force_fresh


# Global variables
_CONFIG_NAME = None
_FORCE_FRESH = False


@hydra.main(version_base=None, config_path="config", config_name="config_attention")
def main(config: DictConfig):
    """
    Main training function with Hydra configuration.
    
    Args:
        config: Hydra configuration object
    """
    global _CONFIG_NAME, _FORCE_FRESH
    
    print("=" * 80)
    print("Attention-LSTM User Identification - PyTorch Lightning + Hydra")
    print("=" * 80)
    
    # Get configuration name from Hydra runtime or global variable
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        config_name = hydra_cfg.job.config_name
    except:
        # Fallback to global variable set during arg parsing
        config_name = _CONFIG_NAME if _CONFIG_NAME else "config_attention"
    
    print(f"\n🔧 Using configuration: {config_name}")
    if _FORCE_FRESH:
        print(f"🆕 Force fresh training mode: Will NOT resume from checkpoint")
    
    # Print configuration
    print("\n📋 Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Set random seeds
    pl.seed_everything(config.seed)
    
    # ========================================================================
    # CACHE SETUP
    # ========================================================================

    cache_paths = get_cache_paths(config, cache_dir=config.cache.cache_dir)
    print_cache_info(cache_paths)
    
    # Check if we can use cached data
    use_cache = check_cache_complete(cache_paths, stages=[1, 2, 3, 4])
    
    # ========================================================================
    # 1. LOAD AND PREPARE DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("[1/7] Loading and preparing data...")
    print("=" * 80)
    
    # Try to load cleaned dataframe from cache
    df_train_clean = load_stage4_cache(cache_paths) if use_cache else None
    
    if df_train_clean is None:
        # Load and clean from scratch
        df_train = load_data(config.data.train_path, data_dir=config.data.data_dir, 
                            size=config.data.size)
        print(f"✓ Loaded {len(df_train)} training samples")
        
        df_train_clean = clean_data(df_train, is_train=True)
        
        # Save to cache
        save_stage4_cache(df_train_clean, cache_paths)
    
    num_users = df_train_clean['user_id'].nunique()
    print(f"✓ Number of unique users: {num_users}")
    
    # Show class distribution
    user_counts = df_train_clean['user_id'].value_counts()
    print(f"✓ Sessions per user: min={user_counts.min()}, "
          f"max={user_counts.max()}, median={user_counts.median():.1f}")
    
    # ========================================================================
    # 2. BUILD VOCABULARY AND TOKENIZE
    # ========================================================================
    print("\n" + "=" * 80)
    print("[2/7] Building vocabulary and tokenizing sequences...")
    print("=" * 80)
    
    # Try to load from cache
    cached_stage1 = load_stage1_cache(cache_paths) if use_cache else None
    
    if cached_stage1 is not None:
        sequences, vocabulary = cached_stage1
    else:
        # Compute from scratch
        sequences, vocabulary = tokenize_actions(df_train_clean, is_train=True)
        
        # Save to cache
        save_stage1_cache(sequences, vocabulary, cache_paths)
    
    vocab_size = len(vocabulary)
    print(f"✓ Vocabulary size: {vocab_size} unique action signatures")
    print(f"✓ Average sequence length: {np.mean([len(s) for s in sequences]):.1f} actions")
    print(f"✓ Max sequence length: {max([len(s) for s in sequences])} actions")
    
    # ========================================================================
    # 3. PREPARE RNN SEQUENCES
    # ========================================================================
    print("\n" + "=" * 80)
    print("[3/7] Preparing RNN sequences...")
    print("=" * 80)
    
    # Try to load from cache
    cached_stage2 = load_stage2_cache(cache_paths) if use_cache else None
    
    if cached_stage2 is not None:
        X_sequences, X_browser, y, user_categories = cached_stage2
    else:
        # Compute from scratch
        X_sequences, X_browser, y, user_categories = prepare_rnn_sequences(
            df_train_clean, vocabulary, max_length=config.data.max_sequence_length, 
            is_train=True
        )
        
        # Save to cache
        save_stage2_cache(X_sequences, X_browser, y, user_categories, cache_paths)
    
    print(f"✓ Sequences shape: {X_sequences.shape}")
    print(f"✓ Browser features shape: {X_browser.shape}")
    print(f"✓ Targets shape: {y.shape}")
    
    # ========================================================================
    # 4. EXTRACT STATISTICAL FEATURES
    # ========================================================================
    print("\n" + "=" * 80)
    print("[4/7] Extracting statistical behavioral features...")
    print("=" * 80)
    
    # Try to load from cache
    X_stat = load_stage3_cache(cache_paths) if use_cache else None
    
    if X_stat is None:
        # Compute from scratch
        X_stat = extract_statistical_features(df_train_clean, action_sequences=sequences)
        
        # Save to cache
        save_stage3_cache(X_stat, cache_paths)
    
    print(f"✓ Statistical features shape: {X_stat.shape}")
    print(f"✓ Number of statistical features: {X_stat.shape[1]}")
    
    # Print feature statistics
    print(f"  • Mean: {X_stat.mean():.4f}")
    print(f"  • Std: {X_stat.std():.4f}")
    print(f"  • Min: {X_stat.min():.4f}")
    print(f"  • Max: {X_stat.max():.4f}")
    
    # ========================================================================
    # 5. CREATE DATA LOADERS
    # ========================================================================
    print("\n" + "=" * 80)
    print("[5/7] Creating data loaders with feature normalization...")
    print("=" * 80)
    
    train_loader, val_loader, train_idx, val_idx, scaler = \
        create_data_loaders_with_features(X_sequences, X_browser, X_stat, y, config)
    
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print(f"✓ Training samples: {len(train_idx)}")
    print(f"✓ Validation samples: {len(val_idx)}")
    
    # ========================================================================
    # 6. INITIALIZE MODEL AND CALLBACKS
    # ========================================================================
    print("\n" + "=" * 80)
    print("[6/7] Initializing Attention-LSTM model...")
    print("=" * 80)
    
    # Create output directory with config name and check for existing checkpoint
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract config identifier (remove 'config_attention_' prefix if present)
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        config_name = hydra_cfg.job.config_name
    except:
        config_name = _CONFIG_NAME if _CONFIG_NAME else "config_attention"
    
    # Extract suffix from config name (e.g., 'large' from 'config_attention_large')
    if config_name.startswith('config_attention_'):
        config_suffix = config_name.replace('config_attention_', '')
        model_prefix = f"attention_lstm_{config_suffix}"
    elif config_name == 'config_attention':
        model_prefix = f"attention_lstm_baseline"
    else:
        model_prefix = f"attention_lstm_{config_name}"
    
    # Check for existing checkpoint to resume from (unless --fresh flag is set)
    resume_checkpoint = None
    existing_dirs = []
    
    if not _FORCE_FRESH and os.path.exists(config.paths.output_dir):
        # Look for directories matching this config
        for dir_name in os.listdir(config.paths.output_dir):
            if dir_name.startswith(model_prefix):
                dir_path = os.path.join(config.paths.output_dir, dir_name)
                if os.path.isdir(dir_path):
                    # Check if last.ckpt exists
                    last_ckpt = os.path.join(dir_path, 'last.ckpt')
                    if os.path.exists(last_ckpt):
                        existing_dirs.append((dir_name, last_ckpt, os.path.getmtime(last_ckpt)))
    
    # If existing checkpoints found, use the most recent one
    if existing_dirs and not _FORCE_FRESH:
        # Sort by modification time (most recent first)
        existing_dirs.sort(key=lambda x: x[2], reverse=True)
        most_recent_dir, resume_checkpoint, _ = existing_dirs[0]
        output_dir = os.path.join(config.paths.output_dir, most_recent_dir)
        model_name = most_recent_dir
        
        print(f"\n🔄 RESUMING FROM CHECKPOINT")
        print(f"✓ Found existing checkpoint: {resume_checkpoint}")
        print(f"✓ Resuming training for: {model_name}")
        print(f"✓ Output directory: {output_dir}")
    else:
        # Create new directory
        model_name = f"{model_prefix}_{timestamp}"
        output_dir = os.path.join(config.paths.output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Starting NEW training")
        print(f"✓ Output directory: {output_dir}")
        print(f"✓ Model identifier: {model_name}")
    
    # Initialize model
    stat_feature_dim = X_stat.shape[1]
    model = AttentionLSTMClassifier(
        config, vocab_size, num_users, stat_feature_dim
    )
    
    # Compute class weights if needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.training.use_class_weights or config.training.use_focal_loss:
        class_weights = compute_class_weights(y[train_idx], num_users, device)
        print(f"✓ Computed class weights for {num_users} users")
        print(f"  • Weight range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
        model.set_loss_function(class_weights)
    else:
        model.set_loss_function(None)
    
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
    # 7. TRAIN MODEL
    # ========================================================================
    # Save hyperparameters/config to YAML in the output directory
    hyperparams_save_path = os.path.join(config.paths.output_dir, "config.yaml")
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    with open(hyperparams_save_path, "w") as f:
        OmegaConf.save(config, f)
    print(f"✓ Saved hyperparameters to {hyperparams_save_path}")
    
    
    print("\n" + "=" * 80)
    print("[7/7] Training Attention-LSTM model...")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  • Max epochs: {config.training.max_epochs}")
    print(f"  • Batch size: {config.data.batch_size}")
    print(f"  • Learning rate: {config.optimizer.lr}")
    print(f"  • Optimizer: {config.optimizer.name}")
    print(f"  • Scheduler: {config.scheduler.name}")
    print(f"  • Loss function: {'Focal Loss' if config.training.use_focal_loss else 'CrossEntropy'}")
    print(f"  • Device: {config.training.accelerator}")
    print(f"  • Bidirectional LSTM: {config.model.bidirectional}")
    print(f"  • Attention dimension: {config.model.attention_dim}")
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
    
    # Train (with resume if checkpoint exists)
    if resume_checkpoint:
        print(f"\n🔄 Resuming training from: {resume_checkpoint}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
    else:
        print(f"\n🆕 Starting fresh training")
        trainer.fit(model, train_loader, val_loader)
    
    # ========================================================================
    # POST-TRAINING: EVALUATION AND RESULTS
    # ========================================================================
    
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Get best metrics
    best_val_score = checkpoint_callback.best_model_score.item() \
        if checkpoint_callback.best_model_score else 0
    print(f"\n✓ Best validation {config.training.checkpoint.monitor}: {best_val_score:.4f}")
    print(f"✓ Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"✓ Last model checkpoint: {checkpoint_callback.last_model_path}")
    
    # Load best model for final evaluation
    best_model = AttentionLSTMClassifier(
        config=config,
        vocab_size=vocab_size,
        num_users=num_users,
        stat_feature_dim=stat_feature_dim
    )
    
    # Initialize criterion with class weights if needed (same as during training)
    if config.training.use_class_weights or config.training.use_focal_loss:
        class_weights = compute_class_weights(y[train_idx], num_users, device)
        best_model.set_loss_function(class_weights)
    
    # Load the checkpoint state dict
    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location=device)
    best_model.load_state_dict(checkpoint['state_dict'])
    
    best_model.eval()
    best_model.to(device)
    
    # Get predictions on train and validation sets
    with torch.no_grad():
        # Training predictions
        train_sequences_tensor = torch.LongTensor(X_sequences[train_idx]).to(device)
        train_stat_tensor = torch.FloatTensor(scaler.transform(X_stat[train_idx])).to(device)
        train_browser_tensor = torch.FloatTensor(X_browser[train_idx]).to(device)
        train_predictions = best_model.predict(
            train_sequences_tensor, train_stat_tensor, train_browser_tensor
        )
        
        # Validation predictions
        val_sequences_tensor = torch.LongTensor(X_sequences[val_idx]).to(device)
        val_stat_tensor = torch.FloatTensor(scaler.transform(X_stat[val_idx])).to(device)
        val_browser_tensor = torch.FloatTensor(X_browser[val_idx]).to(device)
        val_predictions = best_model.predict(
            val_sequences_tensor, val_stat_tensor, val_browser_tensor
        )
        val_predictions_proba = best_model.predict_proba(
            val_sequences_tensor, val_stat_tensor, val_browser_tensor
        )
    
    # Calculate final metrics
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, 
        classification_report
    )
    
    train_metrics = {
        'accuracy': float(accuracy_score(y[train_idx], train_predictions)),
        'f1_weighted': float(f1_score(y[train_idx], train_predictions, average='weighted')),
        'f1_macro': float(f1_score(y[train_idx], train_predictions, average='macro')),
        'precision': float(precision_score(y[train_idx], train_predictions, 
                                          average='weighted', zero_division=0)),
        'recall': float(recall_score(y[train_idx], train_predictions, average='weighted'))
    }
    
    val_metrics = {
        'accuracy': float(accuracy_score(y[val_idx], val_predictions)),
        'f1_weighted': float(f1_score(y[val_idx], val_predictions, average='weighted')),
        'f1_macro': float(f1_score(y[val_idx], val_predictions, average='macro')),
        'precision': float(precision_score(y[val_idx], val_predictions, 
                                          average='weighted', zero_division=0)),
        'recall': float(recall_score(y[val_idx], val_predictions, average='weighted'))
    }
    
    print(f"\n📊 Final Metrics:")
    print(f"\nTraining:")
    for metric, value in train_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    print(f"\nValidation:")
    for metric, value in val_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    # Save results to JSON with model name
    json_path = save_results_to_json(
        config, best_model, train_metrics, val_metrics,
        vocab_size, num_users, checkpoint_callback.best_model_path, output_dir,
        model_name=model_name
    )
    
    # Save artifacts
    artifacts_path = os.path.join(output_dir, 'artifacts')
    os.makedirs(artifacts_path, exist_ok=True)
    
    joblib.dump(vocabulary, os.path.join(artifacts_path, 'vocabulary.joblib'))
    joblib.dump(user_categories, os.path.join(artifacts_path, 'user_categories.joblib'))
    joblib.dump(scaler, os.path.join(artifacts_path, 'stat_scaler.joblib'))
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config, f)
    
    print(f"\n✓ Vocabulary saved to: {artifacts_path}/vocabulary.joblib")
    print(f"✓ User categories saved to: {artifacts_path}/user_categories.joblib")
    print(f"✓ Statistical feature scaler saved to: {artifacts_path}/stat_scaler.joblib")
    print(f"✓ Configuration saved to: {output_dir}/config.yaml")
    
    print("\n" + "=" * 80)
    print("All done! 🎉")
    print("=" * 80)
    print(f"\n💡 To view training logs, run:")
    print(f"   tensorboard --logdir {config.paths.logs_dir}")
    
    return train_metrics, val_metrics


if __name__ == "__main__":
    # Parse custom -c argument and --fresh flag if present
    _CONFIG_NAME, _FORCE_FRESH = parse_custom_args()
    if _CONFIG_NAME:
        print(f"📝 Custom config specified: {_CONFIG_NAME}")
    if _FORCE_FRESH:
        print(f"🆕 Fresh training requested: Will ignore existing checkpoints")
    
    # Run Hydra main
    main()

