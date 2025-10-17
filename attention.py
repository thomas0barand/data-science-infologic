"""
Attention Mechanisms for Sequence Models

This module implements attention mechanisms for RNN-based models,
allowing the model to focus on the most relevant parts of sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    
    This attention mechanism computes a weighted sum of encoder outputs,
    where weights are learned based on the relevance of each timestep.
    
    Reference: "Neural Machine Translation by Jointly Learning to Align and Translate"
    Bahdanau et al., 2015
    """
    
    def __init__(self, hidden_size, attention_dim):
        """
        Initialize attention mechanism.
        
        Args:
            hidden_size (int): Size of hidden states from encoder
            attention_dim (int): Dimension of attention layer
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        # Linear layers for attention computation
        self.W_h = nn.Linear(hidden_size, attention_dim, bias=False)  # Transform hidden states
        self.W_q = nn.Linear(hidden_size, attention_dim, bias=False)  # Transform query (if used)
        self.v = nn.Linear(attention_dim, 1, bias=False)  # Score function
        
    def forward(self, hidden_states, mask=None):
        """
        Compute attention over hidden states.
        
        Args:
            hidden_states (torch.Tensor): Hidden states of shape (batch_size, seq_length, hidden_size)
            mask (torch.Tensor, optional): Mask for padding of shape (batch_size, seq_length)
            
        Returns:
            tuple: (context_vector, attention_weights)
                - context_vector: Weighted sum of hidden states (batch_size, hidden_size)
                - attention_weights: Attention weights (batch_size, seq_length)
        """
        # Project hidden states to attention dimension
        # (batch_size, seq_length, attention_dim)
        projected_hidden = self.W_h(hidden_states)
        
        # Compute attention scores
        # (batch_size, seq_length, 1)
        scores = self.v(torch.tanh(projected_hidden))
        
        # Remove last dimension: (batch_size, seq_length)
        scores = scores.squeeze(-1)
        
        # Apply mask if provided (set padding positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights using softmax
        # (batch_size, seq_length)
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute context vector as weighted sum of hidden states
        # (batch_size, 1, seq_length) @ (batch_size, seq_length, hidden_size)
        # = (batch_size, 1, hidden_size)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states)
        
        # Remove middle dimension: (batch_size, hidden_size)
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism.
    
    Applies multiple attention heads in parallel and combines their outputs.
    This allows the model to attend to different aspects of the sequence.
    """
    
    def __init__(self, hidden_size, num_heads=4, attention_dim=128):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_size (int): Size of hidden states
            num_heads (int): Number of attention heads
            attention_dim (int): Dimension of each attention head
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        
        # Create multiple attention heads
        self.attention_heads = nn.ModuleList([
            BahdanauAttention(hidden_size, attention_dim)
            for _ in range(num_heads)
        ])
        
        # Combine outputs from all heads
        self.combine = nn.Linear(hidden_size * num_heads, hidden_size)
        
    def forward(self, hidden_states, mask=None):
        """
        Apply multi-head attention.
        
        Args:
            hidden_states (torch.Tensor): Hidden states (batch_size, seq_length, hidden_size)
            mask (torch.Tensor, optional): Padding mask (batch_size, seq_length)
            
        Returns:
            tuple: (context_vector, attention_weights_list)
        """
        # Apply each attention head
        contexts = []
        attention_weights_list = []
        
        for attention_head in self.attention_heads:
            context, weights = attention_head(hidden_states, mask)
            contexts.append(context)
            attention_weights_list.append(weights)
        
        # Concatenate contexts from all heads
        # (batch_size, hidden_size * num_heads)
        combined_context = torch.cat(contexts, dim=1)
        
        # Project to original hidden size
        # (batch_size, hidden_size)
        output = self.combine(combined_context)
        
        return output, attention_weights_list


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard negatives.
    This is particularly useful for highly imbalanced classification problems.
    
    Reference: "Focal Loss for Dense Object Detection"
    Lin et al., 2017
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha (torch.Tensor, optional): Class weights of shape (num_classes,)
            gamma (float): Focusing parameter (higher = more focus on hard examples)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Model logits of shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels of shape (batch_size,)
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_padding_mask(sequences, pad_idx=0):
    """
    Create padding mask for sequences.
    
    Args:
        sequences (torch.Tensor): Padded sequences of shape (batch_size, seq_length)
        pad_idx (int): Padding token index (default: 0)
        
    Returns:
        torch.Tensor: Mask of shape (batch_size, seq_length) where 1=valid, 0=padding
    """
    return (sequences != pad_idx).long()

