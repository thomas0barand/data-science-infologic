"""
Attention-LSTM User Classifier with Feature Fusion

This module implements an advanced RNN-based classifier that combines:
- Bidirectional LSTM for sequence modeling
- Attention mechanism to identify discriminative actions
- Statistical feature fusion for enhanced behavioral modeling
- Focal loss for handling class imbalance
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
from torchmetrics import Accuracy, F1Score
from sklearn.preprocessing import StandardScaler

from attention import BahdanauAttention, FocalLoss, create_padding_mask


class AttentionLSTMClassifier(pl.LightningModule):
    """
    Attention-based Bidirectional LSTM for User Classification.
    
    Architecture:
    1. Embedding layer for action tokens
    2. Bidirectional LSTM for sequence encoding
    3. Attention mechanism over LSTM outputs
    4. Feature fusion: [attention_context + statistical_features + browser]
    5. Classification layer with optional focal loss
    """
    
    def __init__(self, config, vocab_size, num_users, stat_feature_dim, browser_dim=4):
        """
        Initialize the Attention-LSTM model.
        
        Args:
            config: Hydra configuration object
            vocab_size (int): Size of action vocabulary
            num_users (int): Number of unique users to classify
            stat_feature_dim (int): Dimension of statistical features
            browser_dim (int): Dimension of browser features (default: 4)
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['config'])
        
        # Store config
        self.config = config
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.stat_feature_dim = stat_feature_dim
        self.browser_dim = browser_dim
        
        # Model architecture parameters
        embedding_dim = config.model.embedding_dim
        hidden_size = config.model.hidden_size
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        bidirectional = config.model.bidirectional
        attention_dim = config.model.attention_dim
        
        # Embedding layer for action tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        self.attention = BahdanauAttention(lstm_output_size, attention_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for statistical features
        self.stat_bn = nn.BatchNorm1d(stat_feature_dim)
        
        # Feature fusion layer
        fusion_input_size = lstm_output_size + stat_feature_dim + browser_dim
        fusion_hidden_size = config.model.get('fusion_hidden_size', 512)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classification layer
        self.classifier = nn.Linear(fusion_hidden_size // 2, num_users)
        
        # Loss function
        if config.training.use_focal_loss:
            # Focal loss will be configured with class weights in training script
            self.criterion = None  # Will be set in setup()
            self.use_focal_loss = True
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.use_focal_loss = False
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_users)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_users)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_users, average='weighted')
        
        # Store validation outputs
        self.validation_step_outputs = []
    
    def set_loss_function(self, class_weights=None):
        """
        Set loss function with optional class weights.
        
        Args:
            class_weights (torch.Tensor): Class weights for imbalanced data
        """
        if self.use_focal_loss:
            gamma = self.config.training.focal_loss.gamma
            self.criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, sequences, statistical_features, browser_features):
        """
        Forward pass through the network.
        
        Args:
            sequences (torch.Tensor): Action sequences (batch_size, seq_length)
            statistical_features (torch.Tensor): Statistical features (batch_size, stat_dim)
            browser_features (torch.Tensor): Browser features (batch_size, browser_dim)
            
        Returns:
            tuple: (logits, attention_weights)
                - logits: Classification logits (batch_size, num_users)
                - attention_weights: Attention weights (batch_size, seq_length)
        """
        batch_size = sequences.size(0)
        
        # 1. Embed sequences: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(sequences)
        embedded = self.dropout(embedded)
        
        # 2. Pass through LSTM
        # lstm_out: (batch_size, seq_length, hidden_size * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 3. Create padding mask for attention
        mask = create_padding_mask(sequences, pad_idx=0)
        
        # 4. Apply attention mechanism
        # context: (batch_size, hidden_size * 2)
        # attention_weights: (batch_size, seq_length)
        context, attention_weights = self.attention(lstm_out, mask=mask)
        context = self.dropout(context)
        
        # 5. Normalize statistical features
        stat_features_norm = self.stat_bn(statistical_features)
        
        # 6. Concatenate all features
        # combined: (batch_size, lstm_output_size + stat_dim + browser_dim)
        combined = torch.cat([context, stat_features_norm, browser_features], dim=1)
        
        # 7. Feature fusion
        fused_features = self.fusion(combined)
        
        # 8. Final classification
        logits = self.classifier(fused_features)
        
        return logits, attention_weights
    
    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        sequences, statistical_features, browser_features, targets = batch
        
        # Forward pass
        logits, _ = self(sequences, statistical_features, browser_features)
        
        # HOTFIX: Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️  Warning: NaN/Inf detected in logits at batch {batch_idx}")
            # Clamp extreme values
            logits = torch.clamp(logits, min=-10, max=10)
        
        loss = self.criterion(logits, targets)
        
        # HOTFIX: Check for NaN loss
        if torch.isnan(loss):
            print(f"⚠️  Warning: NaN loss at batch {batch_idx}, skipping")
            return None
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        sequences, statistical_features, browser_features, targets = batch
        
        # Forward pass
        logits, attention_weights = self(sequences, statistical_features, browser_features)
        
        # HOTFIX: Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️  Warning: NaN/Inf detected in validation logits at batch {batch_idx}")
            logits = torch.clamp(logits, min=-10, max=10)
        
        loss = self.criterion(logits, targets)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, targets)
        f1 = self.val_f1(preds, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': targets
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        optimizer_name = self.config.optimizer.name.lower()
        lr = self.config.optimizer.lr
        weight_decay = self.config.optimizer.weight_decay
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, 
                                       weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Create scheduler
        scheduler_name = self.config.scheduler.name.lower()
        
        if scheduler_name == 'none':
            return optimizer
        
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.config.scheduler.reduce_on_plateau.mode,
                factor=self.config.scheduler.reduce_on_plateau.factor,
                patience=self.config.scheduler.reduce_on_plateau.patience,
                min_lr=self.config.scheduler.reduce_on_plateau.min_lr,
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.config.scheduler.reduce_on_plateau.monitor,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler.cosine.T_max,
                eta_min=self.config.scheduler.cosine.eta_min
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif scheduler_name == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=self.config.scheduler.step.step_size,
                gamma=self.config.scheduler.step.gamma
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def predict(self, sequences, statistical_features, browser_features):
        """
        Generate predictions for sequences.
        
        Args:
            sequences (torch.Tensor): Action sequences
            statistical_features (torch.Tensor): Statistical features
            browser_features (torch.Tensor): Browser features
            
        Returns:
            np.ndarray: Predicted user IDs
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self(sequences, statistical_features, browser_features)
            predictions = torch.argmax(logits, dim=1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, sequences, statistical_features, browser_features):
        """
        Generate probability predictions.
        
        Args:
            sequences (torch.Tensor): Action sequences
            statistical_features (torch.Tensor): Statistical features
            browser_features (torch.Tensor): Browser features
            
        Returns:
            np.ndarray: Predicted probabilities (batch_size, num_users)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self(sequences, statistical_features, browser_features)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.cpu().numpy()
    
    def get_attention_weights(self, sequences, statistical_features, browser_features):
        """
        Get attention weights for sequences (for visualization/analysis).
        
        Args:
            sequences (torch.Tensor): Action sequences
            statistical_features (torch.Tensor): Statistical features
            browser_features (torch.Tensor): Browser features
            
        Returns:
            np.ndarray: Attention weights (batch_size, seq_length)
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self(sequences, statistical_features, browser_features)
            return attention_weights.cpu().numpy()

