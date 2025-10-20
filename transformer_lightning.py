"""
Transformer User Classifier with Local Attention

This module implements a Transformer-based classifier with:
- Local attention windows for efficient long sequence processing
- Positional encoding for sequence order
- Browser feature fusion
- Classification head for user identification
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
from torchmetrics import Accuracy, F1Score
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention is All You Need".
    
    Adds positional information to token embeddings using sine and cosine functions
    of different frequencies.
    """
    
    def __init__(self, d_model, max_len=2000, dropout=0.0):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Embedding dimension
            max_len (int): Maximum sequence length
            dropout (float): Dropout probability (default: 0.0 to avoid early training issues)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Embeddings with positional encoding added
        """
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def create_local_attention_mask(seq_len, window_size, device='cpu'):
    """
    Create attention mask for local attention windows.
    
    Each position can attend to:
    - Itself
    - window_size tokens before and after
    
    Args:
        seq_len (int): Sequence length
        window_size (int): Size of attention window (one-sided)
        device (str): Device to create mask on
        
    Returns:
        torch.Tensor: Float mask of shape (seq_len, seq_len)
                     0.0 = allowed, -inf = masked out
    """
    # Create mask where all positions can initially attend (0.0)
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
    
    # For each position, mask out tokens outside the window
    for i in range(seq_len):
        # Mask everything first
        mask[i, :] = float('-inf')
        # Then unmask the local window
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 0.0
    
    return mask


def create_padding_mask(sequences, pad_idx=0):
    """
    Create padding mask from sequences.
    
    Args:
        sequences (torch.Tensor): Input sequences (batch_size, seq_len)
        pad_idx (int): Padding token index
        
    Returns:
        torch.Tensor: Padding mask (batch_size, seq_len)
                     True = real token, False = padding
    """
    return sequences != pad_idx


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha (torch.Tensor): Class weights
        gamma (float): Focusing parameter (default: 2.0)
    """
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Logits (batch_size, num_classes)
            targets (torch.Tensor): Target labels (batch_size,)
            
        Returns:
            torch.Tensor: Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()
        return focal_loss


class TransformerUserClassifier(pl.LightningModule):
    """
    Transformer-based User Classifier with Local Attention.
    
    Architecture:
    1. Token embedding + positional encoding
    2. Transformer encoder with local attention
    3. Global pooling (CLS token or mean pooling)
    4. Feature fusion with browser features
    5. Classification head
    """
    
    def __init__(self, config, vocab_size, num_users, browser_dim=4):
        """
        Initialize the Transformer model.
        
        Args:
            config: Hydra configuration object
            vocab_size (int): Size of action vocabulary
            num_users (int): Number of unique users to classify
            browser_dim (int): Dimension of browser features (default: 4)
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['config'])
        
        # Store config
        self.config = config
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.browser_dim = browser_dim
        
        # Model architecture parameters
        d_model = config.model.d_model
        nhead = config.model.nhead
        num_layers = config.model.num_layers
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.dropout
        self.local_window = config.model.get('local_attention_window', 128)
        self.pooling_method = config.model.get('pooling', 'cls')
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # CLS token (if using CLS pooling)
        if self.pooling_method == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding (no dropout here, dropout is applied in transformer layers)
        max_len = config.data.max_sequence_length + (1 if self.pooling_method == 'cls' else 0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head with feature fusion
        fusion_input_size = d_model + browser_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_users)
        )
        
        # Loss function
        if self.config.training.use_focal_loss:
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with proper scaling for numerical stability."""
        # Initialize embeddings with scaled normal distribution
        # Use smaller std for better stability
        d_model = self.config.model.d_model
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        # Set padding token embedding to zero
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        
        # Initialize CLS token if present with very small values
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, mean=0, std=0.01)
        
        # Initialize classifier with careful scaling
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                # Use smaller gain for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
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
    
    def forward(self, sequences, browser_features):
        """
        Forward pass through the network.
        
        Args:
            sequences (torch.Tensor): Action sequences (batch_size, seq_length)
            browser_features (torch.Tensor): Browser features (batch_size, browser_dim)
            
        Returns:
            torch.Tensor: Classification logits (batch_size, num_users)
        """
        batch_size, seq_len = sequences.size()
        device = sequences.device
        
        # 1. Embed sequences: (batch_size, seq_length) -> (batch_size, seq_length, d_model)
        embedded = self.embedding(sequences)
        # Scale embeddings more conservatively to prevent overflow
        # Original paper uses sqrt(d_model), but we use a smaller factor for stability
        scale_factor = min(math.sqrt(self.config.model.d_model), 4.0)
        embedded = embedded * scale_factor
        
        # 2. Add CLS token if using CLS pooling
        if self.pooling_method == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embedded = torch.cat([cls_tokens, embedded], dim=1)
            seq_len += 1
        
        # 3. Add positional encoding
        embedded = self.pos_encoder(embedded)
        
        # 4. Create attention masks
        # Padding mask: True for real tokens, False for padding
        if self.pooling_method == 'cls':
            # Add True for CLS token at the beginning
            padding_mask = create_padding_mask(sequences, pad_idx=0)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            src_key_padding_mask = create_padding_mask(sequences, pad_idx=0)
        
        # For PyTorch Transformer: True values are ignored, False values are used
        # So we need to invert our mask
        src_key_padding_mask = ~src_key_padding_mask
        
        # Local attention mask (already in correct format: 0.0 = allowed, -inf = masked)
        attn_mask = create_local_attention_mask(seq_len, self.local_window, device=device)
        
        # 5. Pass through transformer encoder
        transformer_out = self.transformer_encoder(
            embedded,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 6. Global pooling
        if self.pooling_method == 'cls':
            # Use CLS token output
            pooled = transformer_out[:, 0, :]
        else:
            # Mean pooling over non-padding tokens with numerical stability
            mask_expanded = (~src_key_padding_mask).unsqueeze(-1).expand(transformer_out.size())
            sum_out = (transformer_out * mask_expanded.float()).sum(1)
            count = mask_expanded.sum(1).clamp(min=1e-8)  # Prevent division by zero
            pooled = sum_out / count
        
        # Check for NaN/Inf after pooling
        if torch.isnan(pooled).any() or torch.isinf(pooled).any():
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 7. Concatenate with browser features
        combined = torch.cat([pooled, browser_features], dim=1)
        
        # 8. Classification
        logits = self.classifier(combined)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        sequences, browser_features, targets = batch
        
        # Validate inputs for NaN/Inf
        if torch.isnan(browser_features).any() or torch.isinf(browser_features).any():
            print(f"⚠️  Warning: NaN/Inf in browser features at batch {batch_idx}")
            browser_features = torch.nan_to_num(browser_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Forward pass
        logits = self(sequences, browser_features)
        
        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️  Warning: NaN/Inf detected in training logits at batch {batch_idx}")
            print(f"   - NaN count: {torch.isnan(logits).sum().item()}")
            print(f"   - Inf count: {torch.isinf(logits).sum().item()}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        loss = self.criterion(logits, targets)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  Warning: NaN/Inf loss at batch {batch_idx}, skipping")
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
        sequences, browser_features, targets = batch
        
        # Validate inputs for NaN/Inf
        if torch.isnan(browser_features).any() or torch.isinf(browser_features).any():
            print(f"⚠️  Warning: NaN/Inf in browser features at validation batch {batch_idx}")
            browser_features = torch.nan_to_num(browser_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Forward pass
        logits = self(sequences, browser_features)
        
        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️  Warning: NaN/Inf detected in validation logits at batch {batch_idx}")
            print(f"   - NaN count: {torch.isnan(logits).sum().item()}")
            print(f"   - Inf count: {torch.isinf(logits).sum().item()}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        loss = self.criterion(logits, targets)
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  Warning: NaN/Inf loss at validation batch {batch_idx}")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
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
        Generate probability predictions.
        
        Args:
            sequences (torch.Tensor): Action sequences
            browser_features (torch.Tensor): Browser features
            
        Returns:
            np.ndarray: Predicted probabilities (batch_size, num_users)
        """
        self.eval()
        with torch.no_grad():
            logits = self(sequences, browser_features)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.cpu().numpy()

