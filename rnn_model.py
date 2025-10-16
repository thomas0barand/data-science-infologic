"""
Simple RNN Model for User Classification

This module implements a Simple RNN architecture using PyTorch
to predict users from sequential action data.
"""

import torch
import torch.nn as nn
import numpy as np


class SimpleRNNUserClassifier(nn.Module):
    """
    Simple RNN model for user classification from action sequences.
    
    Architecture:
        - Embedding layer: vocab_size -> embedding_dim
        - Simple RNN layer: embedding_dim -> hidden_size
        - Dropout layer
        - Concatenate RNN output with browser features
        - Dense layer: (hidden_size + browser_dim) -> num_users
    """
    
    def __init__(self, vocab_size, num_users, embedding_dim=128, 
                 hidden_size=256, browser_dim=4, dropout=0.3):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size (int): Size of action vocabulary
            num_users (int): Number of unique users to classify
            embedding_dim (int): Dimension of embedding layer (default: 128)
            hidden_size (int): Size of RNN hidden state (default: 256)
            browser_dim (int): Dimension of browser features (default: 4)
            dropout (float): Dropout rate (default: 0.3)
        """
        super(SimpleRNNUserClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.browser_dim = browser_dim
        
        # Embedding layer for action tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Simple RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Final classification layer (RNN output + browser features)
        self.fc = nn.Linear(hidden_size + browser_dim, num_users)
    
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
        
        # Pass through RNN: output shape (batch_size, seq_length, hidden_size)
        # We only need the last hidden state
        rnn_out, hidden = self.rnn(embedded)
        
        # Take the last output (or use hidden state)
        # hidden shape: (1, batch_size, hidden_size)
        last_hidden = hidden.squeeze(0)  # (batch_size, hidden_size)
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Concatenate with browser features
        combined = torch.cat([last_hidden, browser_features], dim=1)
        
        # Final classification
        logits = self.fc(combined)
        
        return logits
    
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
            logits = self.forward(sequences, browser_features)
            predictions = torch.argmax(logits, dim=1)
            # Convert to list first to avoid numpy compatibility issues
            return np.array(predictions.cpu().tolist())
    
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
            logits = self.forward(sequences, browser_features)
            probabilities = torch.softmax(logits, dim=1)
            # Convert to list first to avoid numpy compatibility issues
            return np.array(probabilities.cpu().tolist())


def create_data_loaders(sequences, browser_features, targets, batch_size=32, 
                        train_split=0.8, random_state=42):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        sequences (np.ndarray): Action sequences
        browser_features (np.ndarray): Browser features
        targets (np.ndarray): Target user IDs
        batch_size (int): Batch size for training
        train_split (float): Proportion of data for training
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, train_indices, val_indices)
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    from collections import Counter
    
    # Check if stratified splitting is possible
    # (all classes must have at least 2 samples)
    class_counts = Counter(targets)
    min_class_count = min(class_counts.values())
    use_stratify = min_class_count >= 2
    
    if not use_stratify:
        print(f"⚠️  Warning: {sum(1 for c in class_counts.values() if c < 2)} classes have < 2 samples. Using non-stratified split.")
    
    # Split data
    indices = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(
        indices, test_size=(1 - train_split), random_state=random_state,
        stratify=targets if use_stratify else None
    )
    
    # Create training dataset
    train_sequences = torch.LongTensor(sequences[train_idx])
    train_browser = torch.FloatTensor(browser_features[train_idx])
    train_targets = torch.LongTensor(targets[train_idx])
    train_dataset = TensorDataset(train_sequences, train_browser, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation dataset
    val_sequences = torch.LongTensor(sequences[val_idx])
    val_browser = torch.FloatTensor(browser_features[val_idx])
    val_targets = torch.LongTensor(targets[val_idx])
    val_dataset = TensorDataset(val_sequences, val_browser, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_idx, val_idx


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: RNN model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, browser_features, targets in train_loader:
        sequences = sequences.to(device)
        browser_features = browser_features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences, browser_features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate model on validation data.
    
    Args:
        model: RNN model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        tuple: (average_loss, accuracy, predictions, targets)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, browser_features, targets in val_loader:
            sequences = sequences.to(device)
            browser_features = browser_features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(sequences, browser_features)
            loss = criterion(outputs, targets)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Convert to list to avoid numpy compatibility issues
            all_predictions.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets)

