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
        

        # ============================================================================
        # DEEP CLASSIFICATION HEAD with Advanced Architecture
        # ============================================================================
        # Architecture: Multi-scale residual blocks + SE attention + progressive reduction
        # Recommended fusion_hidden_size: 512-1024 for best performance
        
        classifier_input_size = fusion_hidden_size // 2
        
        # Define intermediate dimensions with progressive reduction
        # This creates a funnel architecture: wide -> narrow -> output
        dim1 = classifier_input_size  # e.g., 256 if fusion_hidden_size=512
        dim2 = max(dim1, 256)  # Ensure at least 256 for expressiveness
        dim3 = dim2 // 2  # e.g., 128
        dim4 = max(dim3, 128)  # Ensure at least 128
        
        # Store dimensions for weight initialization
        self.classifier_dims = [dim1, dim2, dim3, dim4]
        
        # ------------------------------------------------------------------------
        # Block 1: Input expansion with residual connection
        # ------------------------------------------------------------------------
        self.classifier_block1 = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(dim1, dim2),
                nn.LayerNorm(dim2),
                nn.GELU(),
                nn.Dropout(dropout * 1.2),
                nn.Linear(dim2, dim2),
                nn.LayerNorm(dim2),
                nn.GELU(),
                nn.Dropout(dropout * 1.2),
            ),
            'residual': nn.Linear(dim1, dim2) if dim1 != dim2 else nn.Identity(),
            'se_attention': self._create_se_block(dim2, reduction=8)
        })
        
        # ------------------------------------------------------------------------
        # Block 2: Deep residual block with attention
        # ------------------------------------------------------------------------
        self.classifier_block2 = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(dim2, dim2),
                nn.LayerNorm(dim2),
                nn.GELU(),
                nn.Dropout(dropout * 1.3),
                nn.Linear(dim2, dim2),
                nn.LayerNorm(dim2),
                nn.GELU(),
                nn.Dropout(dropout * 1.3),
            ),
            'se_attention': self._create_se_block(dim2, reduction=8)
        })
        
        # ------------------------------------------------------------------------
        # Block 3: Dimensionality reduction block
        # ------------------------------------------------------------------------
        self.classifier_block3 = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(dim2, dim3),
                nn.LayerNorm(dim3),
                nn.GELU(),
                nn.Dropout(dropout * 1.4),
                nn.Linear(dim3, dim3),
                nn.LayerNorm(dim3),
                nn.GELU(),
                nn.Dropout(dropout * 1.4),
            ),
            'residual': nn.Linear(dim2, dim3),
            'se_attention': self._create_se_block(dim3, reduction=4)
        })
        
        # ------------------------------------------------------------------------
        # Block 4: Further refinement
        # ------------------------------------------------------------------------
        self.classifier_block4 = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(dim3, dim4),
                nn.LayerNorm(dim4),
                nn.GELU(),
                nn.Dropout(dropout * 1.5),
                nn.Linear(dim4, dim4),
                nn.LayerNorm(dim4),
                nn.GELU(),
                nn.Dropout(dropout * 1.5),
            ),
            'residual': nn.Linear(dim3, dim4) if dim3 != dim4 else nn.Identity(),
            'se_attention': self._create_se_block(dim4, reduction=4)
        })
        
        # ------------------------------------------------------------------------
        # Block 5: Final refinement before output
        # ------------------------------------------------------------------------
        self.classifier_block5 = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(dim4, dim4),
                nn.LayerNorm(dim4),
                nn.GELU(),
                nn.Dropout(dropout * 1.6),
            ),
            'se_attention': self._create_se_block(dim4, reduction=4)
        })
        
        # ------------------------------------------------------------------------
        # Final projection layer
        # ------------------------------------------------------------------------
        self.classifier_output = nn.Sequential(
            nn.Linear(dim4, dim4 // 2),
            nn.LayerNorm(dim4 // 2),
            nn.GELU(),
            nn.Dropout(dropout * 1.8),
            nn.Linear(dim4 // 2, num_users)
        )
        
        # Stochastic depth probabilities for each block (optional, for training stability)
        self.drop_path_probs = [0.0, 0.05, 0.1, 0.15, 0.2]
        
        # Initialize all classifier weights
        self._init_classifier_weights()
    
    def _create_se_block(self, channels, reduction=8):
        """
        Create Squeeze-and-Excitation block for feature recalibration.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Reduction ratio for bottleneck
            
        Returns:
            nn.Sequential: SE block
        """
        return nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )
    
    def _init_classifier_weights(self):
        """Initialize classifier weights using Xavier/He initialization."""
        # Initialize all blocks
        for block_name in ['classifier_block1', 'classifier_block2', 'classifier_block3', 
                          'classifier_block4', 'classifier_block5']:
            if hasattr(self, block_name):
                block = getattr(self, block_name)
                for key, module in block.items():
                    if isinstance(module, nn.Sequential):
                        for m in module:
                            if isinstance(m, nn.Linear):
                                nn.init.xavier_uniform_(m.weight, gain=1.0)
                                if m.bias is not None:
                                    nn.init.constant_(m.bias, 0)
                    elif isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=1.0)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
        
        # Initialize output projection with smaller weights for stability
        for module in self.classifier_output:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Loss function
        if self.config.training.use_focal_loss:
            # Focal loss will be configured with class weights in training script
            self.criterion = None  # Will be set in setup()
            self.use_focal_loss = True
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.use_focal_loss = False
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_users)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_users)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_users, average='weighted')
        
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
        
        # ========================================================================
        # 8. DEEP CLASSIFICATION HEAD with Multi-scale Residual + SE Attention
        # ========================================================================
        x = fused_features
        
        # Block 1: Input expansion + SE attention
        residual = self.classifier_block1['residual'](x)
        x = self.classifier_block1['main'](x)
        se_weight = self.classifier_block1['se_attention'](x)
        x = x * se_weight  # Apply SE attention
        x = x + residual  # Residual connection
        
        # Block 2: Deep refinement + SE attention (same dimension)
        residual = x
        x = self.classifier_block2['main'](x)
        se_weight = self.classifier_block2['se_attention'](x)
        x = x * se_weight
        x = x + residual
        
        # Block 3: Dimensionality reduction + SE attention
        residual = self.classifier_block3['residual'](x)
        x = self.classifier_block3['main'](x)
        se_weight = self.classifier_block3['se_attention'](x)
        x = x * se_weight
        x = x + residual
        
        # Block 4: Further refinement + SE attention
        residual = self.classifier_block4['residual'](x)
        x = self.classifier_block4['main'](x)
        se_weight = self.classifier_block4['se_attention'](x)
        x = x * se_weight
        x = x + residual
        
        # Block 5: Final refinement + SE attention (same dimension)
        residual = x
        x = self.classifier_block5['main'](x)
        se_weight = self.classifier_block5['se_attention'](x)
        x = x * se_weight
        x = x + residual
        
        # Final projection to class logits
        logits = self.classifier_output(x)
        
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
                min_lr=self.config.scheduler.reduce_on_plateau.min_lr
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
            # Apply softmax to convert logits to probabilities
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

