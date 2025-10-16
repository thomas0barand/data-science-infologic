# RNN Implementation for User Identification

This document describes the Simple RNN implementation for predicting users from action sequences.

## Overview

The RNN implementation uses a Simple Recurrent Neural Network to classify users based on their sequential action patterns. The model processes sequences of up to 1000 actions and combines them with browser information for final classification.

## Architecture

### Model Structure
```
Input: Action Sequence (1000 tokens) + Browser Features (4 dims)
    ↓
Embedding Layer (vocab_size → 128)
    ↓
Simple RNN Layer (128 → 256)
    ↓
Dropout (0.3)
    ↓
Concatenate with Browser Features
    ↓
Dense Layer (260 → num_users)
    ↓
Softmax → User Prediction
```

### Key Parameters
- **Vocabulary Size**: ~50-100 unique action types
- **Embedding Dimension**: 128
- **Hidden Size**: 256
- **Max Sequence Length**: 1000 actions
- **Browser Dimension**: 4 (one-hot encoded)
- **Dropout**: 0.3

## Data Preprocessing

### Action Tokenization
Actions are parsed to extract base action types:
- `"Création d'un écran(infologic.core...)"` → `"Création d'un écran"`
- `"Exécution d'un bouton<DEF_03/24>"` → `"Exécution d'un bouton"`
- `"Saisie dans un champ$JCP$"` → `"Saisie dans un champ"`

### Sequence Handling
1. **Filter**: Remove time markers (t5, t10, t15, etc.)
2. **Truncate**: Keep last 1000 actions if sequence is longer
3. **Pad**: Add padding tokens (ID: 0) if sequence is shorter
4. **Encode**: Convert action strings to integer IDs using vocabulary

### Browser Encoding
One-hot encode browser types:
- Firefox: [1, 0, 0, 0]
- Google Chrome: [0, 1, 0, 0]
- Microsoft Edge: [0, 0, 1, 0]
- Opera: [0, 0, 0, 1]

## Usage

### Training the Model

```bash
python train_rnn.py
```

This will:
1. Load and preprocess training data
2. Build vocabulary from action sequences
3. Train the RNN model for 25 epochs
4. Save model, vocabulary, and metrics to `results/` directory

**Training Parameters:**
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 25
- Train/Val split: 80/20

### Making Predictions

```bash
python predict_rnn.py --model_dir results/models/simple_rnn_YYYYMMDD_HHMMSS --output submission_rnn.csv
```

This will:
1. Load the trained model and vocabulary
2. Process test data
3. Generate predictions
4. Save results to CSV file

## Files Created

### Core Files
- **`utils.py`** (modified): Added RNN preprocessing functions
  - `parse_action_string()`: Parse action strings
  - `tokenize_actions()`: Build vocabulary and tokenize sequences
  - `encode_browser()`: One-hot encode browser features
  - `prepare_rnn_sequences()`: Complete preprocessing pipeline

- **`rnn_model.py`**: RNN model architecture
  - `SimpleRNNUserClassifier`: Main model class
  - `create_data_loaders()`: Create PyTorch DataLoaders
  - `train_epoch()`: Training loop for one epoch
  - `evaluate()`: Evaluation function

- **`train_rnn.py`**: Training script
  - Complete training pipeline
  - Model saving and metrics tracking
  - Integration with ModelManager

- **`predict_rnn.py`**: Prediction script
  - Load trained model
  - Process test data
  - Generate submission file

### Output Files (in `results/`)
```
results/
├── models/
│   └── simple_rnn_YYYYMMDD_HHMMSS/
│       ├── simple_rnn_YYYYMMDD_HHMMSS_model.pt
│       ├── simple_rnn_YYYYMMDD_HHMMSS_vocabulary.joblib
│       ├── simple_rnn_YYYYMMDD_HHMMSS_categories.joblib
│       ├── simple_rnn_YYYYMMDD_HHMMSS_history.joblib
│       └── simple_rnn_YYYYMMDD_HHMMSS_metadata.json
└── metrics/
    └── simple_rnn_YYYYMMDD_HHMMSS_metrics.json
```

## Expected Performance

### Baseline Expectations
- **Training Accuracy**: 60-75%
- **Validation Accuracy**: 55-70%
- **F1-Score (weighted)**: 55-70%

### Comparison with Logistic Regression
- Logistic Regression: ~70% validation accuracy
- RNN: Expected to be comparable or slightly better due to sequential modeling

### Advantages
- Captures temporal patterns in user behavior
- Learns action sequence dependencies
- Can identify characteristic action patterns per user

### Limitations
- Requires more training time than feature-based models
- More parameters to tune
- Needs sufficient data per user for good performance

## Training Tips

### If accuracy is low:
1. **Increase epochs**: Try 30-40 epochs
2. **Adjust learning rate**: Try 0.0005 or 0.002
3. **Increase hidden size**: Try 512 instead of 256
4. **Add more layers**: Stack multiple RNN layers
5. **Try LSTM/GRU**: Better for long sequences

### If overfitting occurs:
1. **Increase dropout**: Try 0.4 or 0.5
2. **Add regularization**: L2 weight decay
3. **Reduce model size**: Smaller hidden size
4. **Early stopping**: Stop when val loss increases

### For better performance:
1. **Ensemble**: Combine RNN with Logistic Regression
2. **Attention mechanism**: Add attention to focus on important actions
3. **Bidirectional RNN**: Process sequences in both directions
4. **Data augmentation**: Create variations of sequences

## Dependencies

The implementation requires:
- PyTorch (for RNN model)
- NumPy (for array operations)
- Pandas (for data handling)
- scikit-learn (for train/val split and metrics)
- joblib (for saving artifacts)

Install with:
```bash
pip install torch numpy pandas scikit-learn joblib
```

## Integration with Existing System

The RNN implementation integrates seamlessly with the existing model management system:
- Uses the same `ModelManager` for saving models and metrics
- Follows the same naming conventions
- Stores results in the same `results/` directory structure
- Compatible with `view_results.py` for comparing models

## Next Steps

1. **Train the model**: Run `python train_rnn.py`
2. **Evaluate results**: Compare with Logistic Regression
3. **Generate predictions**: Use `predict_rnn.py` for test data
4. **Experiment**: Try different hyperparameters
5. **Ensemble**: Combine with other models for better performance

