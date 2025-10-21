"""
Simpler Attention-LSTM Classifier (for backward compatibility with older checkpoints)

This version matches the architecture of checkpoints trained with the simpler classifier.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from attention import BahdanauAttention, create_padding_mask


class AttentionLSTMClassifierSimple(pl.LightningModule):
    """
    Simplified Attention-LSTM Classifier (matches older checkpoint architecture).
    
    Architecture:
    1. Embedding layer for action tokens
    2. LSTM for sequence encoding (unidirectional or bidirectional)
    3. Attention mechanism over LSTM outputs
    4. Feature fusion: [attention_context + statistical_features + browser]
    5. Simple classification layers
    """
    
    def __init__(self, vocab_size, num_users, stat_feature_dim, browser_dim=4,
                 embedding_dim=64, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False, attention_dim=128, fusion_hidden_size=256):
        """
        Initialize the simplified Attention-LSTM model.
        
        Args:
            vocab_size (int): Size of action vocabulary
            num_users (int): Number of unique users to classify
            stat_feature_dim (int): Dimension of statistical features
            browser_dim (int): Dimension of browser features (default: 4)
            embedding_dim (int): Embedding dimension (default: 64)
            hidden_size (int): LSTM hidden size (default: 128)
            num_layers (int): Number of LSTM layers (default: 1)
            dropout (float): Dropout rate (default: 0.2)
            bidirectional (bool): Use bidirectional LSTM (default: False)
            attention_dim (int): Attention dimension (default: 128)
            fusion_hidden_size (int): Fusion layer hidden size (default: 256)
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.stat_feature_dim = stat_feature_dim
        self.browser_dim = browser_dim
        
        # Embedding layer for action tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM (can be unidirectional or bidirectional)
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
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple classification layers (2 blocks)
        classifier_input_size = fusion_hidden_size // 2
        
        self.classifier_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(classifier_input_size, classifier_input_size),
                nn.LayerNorm(classifier_input_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(classifier_input_size, classifier_input_size),
                nn.LayerNorm(classifier_input_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        # Output layer
        self.classifier_output = nn.Linear(classifier_input_size, num_users)
    
    def forward(self, sequences, statistical_features, browser_features):
        """
        Forward pass through the network.
        
        Args:
            sequences (torch.Tensor): Action sequences (batch_size, seq_length)
            statistical_features (torch.Tensor): Statistical features (batch_size, stat_dim)
            browser_features (torch.Tensor): Browser features (batch_size, browser_dim)
            
        Returns:
            tuple: (logits, attention_weights)
        """
        batch_size = sequences.size(0)
        
        # 1. Embed sequences
        embedded = self.embedding(sequences)
        embedded = self.dropout(embedded)
        
        # 2. Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 3. Create padding mask for attention
        mask = create_padding_mask(sequences, pad_idx=0)
        
        # 4. Apply attention mechanism
        context, attention_weights = self.attention(lstm_out, mask=mask)
        context = self.dropout(context)
        
        # 5. Normalize statistical features
        stat_features_norm = self.stat_bn(statistical_features)
        
        # 6. Concatenate all features
        combined = torch.cat([context, stat_features_norm, browser_features], dim=1)
        
        # 7. Feature fusion
        fused_features = self.fusion(combined)
        
        # 8. Pass through classifier layers
        x = fused_features
        for layer in self.classifier_layers:
            x = layer(x)
        
        # 9. Output layer
        logits = self.classifier_output(x)
        
        return logits, attention_weights
    
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

