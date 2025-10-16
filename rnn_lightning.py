"""
PyTorch Lightning Module for RNN User Classification

This module implements a Lightning-based RNN for user identification
with integrated training, validation, and metrics logging.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
from torchmetrics import Accuracy, F1Score


class RNNUserClassifierLightning(pl.LightningModule):
    """
    PyTorch Lightning module for RNN-based user classification.
    
    Integrates model definition, training loop, validation, optimizers,
    and learning rate scheduling into a single cohesive module.
    """
    
    def __init__(self, config, vocab_size, num_users, browser_dim=4):
        """
        Initialize the Lightning RNN model.
        
        Args:
            config: Hydra configuration object
            vocab_size (int): Size of action vocabulary
            num_users (int): Number of unique users to classify
            browser_dim (int): Dimension of browser features (default: 4)
        """
        super().__init__()
        
        # Save hyperparameters to checkpoint
        self.save_hyperparameters(ignore=['config'])
        
        # Store config
        self.config = config
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.browser_dim = browser_dim
        
        # Model architecture
        embedding_dim = config.model.embedding_dim
        hidden_size = config.model.hidden_size
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        bidirectional = config.model.bidirectional
        
        # Embedding layer for action tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer (supports multiple layers and bidirectional)
        self.rnn = nn.RNN(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Calculate RNN output size
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Final classification layer (RNN output + browser features)
        self.fc = nn.Linear(rnn_output_size + browser_dim, num_users)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_users)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_users)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_users, average='weighted')
        
        # Store predictions for epoch-end metrics
        self.validation_step_outputs = []
    
    def forward(self, sequences, browser_features):
        """
        Forward pass through the network.
        
        Args:
            sequences (torch.Tensor): Action sequences of shape (batch_size, seq_length)
            browser_features (torch.Tensor): Browser features of shape (batch_size, browser_dim)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_users)
        """
        # Embed sequences: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(sequences)
        
        # Pass through RNN
        rnn_out, hidden = self.rnn(embedded)
        
        # Extract last hidden state
        # For bidirectional, concatenate forward and backward final states
        if self.config.model.bidirectional:
            # hidden shape: (num_layers * 2, batch_size, hidden_size)
            # Take last layer's forward and backward hidden states
            num_layers = self.config.model.num_layers
            forward_hidden = hidden[2 * (num_layers - 1)]
            backward_hidden = hidden[2 * (num_layers - 1) + 1]
            last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # hidden shape: (num_layers, batch_size, hidden_size)
            last_hidden = hidden[-1]  # Take last layer
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Concatenate with browser features
        combined = torch.cat([last_hidden, browser_features], dim=1)
        
        # Final classification
        logits = self.fc(combined)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.
        
        Args:
            batch: Tuple of (sequences, browser_features, targets)
            batch_idx: Index of the batch
            
        Returns:
            torch.Tensor: Loss value
        """
        sequences, browser_features, targets = batch
        
        # Forward pass
        logits = self(sequences, browser_features)
        loss = self.criterion(logits, targets)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.
        
        Args:
            batch: Tuple of (sequences, browser_features, targets)
            batch_idx: Index of the batch
            
        Returns:
            dict: Dictionary containing loss and predictions
        """
        sequences, browser_features, targets = batch
        
        # Forward pass
        logits = self(sequences, browser_features)
        loss = self.criterion(logits, targets)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, targets)
        f1 = self.val_f1(preds, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end processing
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': targets
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch.
        Can be used for additional metrics computation.
        """
        # Clear stored outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            dict: Dictionary with optimizer and optionally scheduler
        """
        # Create optimizer
        optimizer_name = self.config.optimizer.name.lower()
        lr = self.config.optimizer.lr
        weight_decay = self.config.optimizer.weight_decay
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
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
    
    def predict(self, sequences, browser_features):
        """
        Generate predictions for sequences.
        
        Args:
            sequences (torch.Tensor): Action sequences
            browser_features (torch.Tensor): Browser features
            
        Returns:
            np.ndarray: Predicted user IDs
        """
        self.eval()
        with torch.no_grad():
            logits = self(sequences, browser_features)
            predictions = torch.argmax(logits, dim=1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, sequences, browser_features):
        """
        Generate probability predictions for sequences.
        
        Args:
            sequences (torch.Tensor): Action sequences
            browser_features (torch.Tensor): Browser features
            
        Returns:
            np.ndarray: Predicted probabilities of shape (batch_size, num_users)
        """
        self.eval()
        with torch.no_grad():
            logits = self(sequences, browser_features)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.cpu().numpy()

