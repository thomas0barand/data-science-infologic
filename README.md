# Data Science Infologic - Thomas Branch

## Description

This branch focuses on implementing and training two different machine learning approaches for user identification:

1. **Logistic Regression**: A traditional statistical approach using handcrafted features extracted from user session data
2. **Attention-LSTM Architecture**: A deep learning approach combining LSTM networks with attention mechanisms to capture sequential patterns in user behavior

The project implements a complete machine learning pipeline from data preprocessing to model training and inference for identifying users based on their interaction traces with the Copilote application.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/thomas0barand/data-science-infologic.git
cd data-science-infologic
git checkout thomas
```

### Prerequisites

- Python >= 3.12, < 3.14
- Poetry (recommended) or pip

### Option 1: Using Poetry

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies and activate environment**:
   ```bash
   poetry install
   poetry shell
   ```

### Option 2: Using requirements.txt

1. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```
data-science-infologic/
├── config/                    # Hydra configuration files
│   ├── config_attention_lstm.yaml
│   ├── config_attention_ultra_light.yaml
│   ├── config_attention_light.yaml
│   └── ... (other config variants)
├── training/                  # Training scripts
│   ├── train_logistic.py     # Logistic Regression training
│   ├── train_attention_lstm.py  # Attention-LSTM training
│   ├── inference_attention_lstm.py  # Inference script
│   ├── model/                # Model architectures
│   │   └── attention_lstm.py
│   └── utils/                # Utilities
│       ├── utils.py          # Data loading, feature extraction
│       └── cache_utils.py    # Caching system
├── data/                     # Dataset files (not in repo)
│   ├── train.csv
│   └── test.csv
├── results/                  # Training results
│   ├── models/              # Saved models
│   └── metrics/              # Training metrics
├── cache/                    # Preprocessed data cache
├── logs/                     # TensorBoard logs
└── pyproject.toml            # Poetry configuration
```

---

## Training

### Logistic Regression Training

Logistic Regression uses handcrafted statistical features extracted from user sessions.

**Launch training:**
```bash
# From project root
python training/train_logistic.py
```

**What it does:**
1. Loads and cleans training data from `data/train.csv`
2. Extracts statistical features (action counts, ratios, diversity metrics, etc.)
3. Splits data into train/validation sets (80/20)
4. Trains a Logistic Regression model with feature scaling
5. Evaluates on validation set and saves results

**Results location:**
- Model: `results/models/logistic_regression_YYYYMMDD_HHMMSS/`
- Metrics: `results/metrics/logistic_regression_YYYYMMDD_HHMMSS_metrics.json`

### Attention-LSTM Training

The Attention-LSTM model combines sequence learning with statistical features for better user identification.

**Launch training with default config:**
```bash
# From project root
python training/train_attention_lstm.py --config-name config_attention_lstm
```

**Or using short syntax:**
```bash
python training/train_attention_lstm.py -c config_attention_lstm
```

**What it does:**
1. Loads and preprocesses training data
2. Tokenizes action sequences and builds vocabulary
3. Extracts statistical features
4. Prepares sequences for RNN (pads/truncates to max_length)
5. Trains Attention-LSTM model with PyTorch Lightning
6. Saves checkpoints, artifacts (vocabulary, scaler), and metrics

**Key features:**
- **Caching system**: Preprocessed data is cached to speed up subsequent runs
- **Auto-resume**: Automatically resumes from last checkpoint if training is interrupted
- **Force fresh training**: Use `--fresh` flag to ignore existing checkpoints

**Results location:**
- Model checkpoints: `results/models/attention_lstm_<config_suffix>_YYYYMMDD_HHMMSS/`
- Artifacts: `results/models/.../artifacts/` (vocabulary, scaler, user categories)
- Metrics: `results/metrics/attention_lstm_<config_suffix>_YYYYMMDD_HHMMSS_metrics.json`
- TensorBoard logs: `logs/attention_lstm_<config_suffix>_YYYYMMDD_HHMMSS/`

---

## Changing Training Configuration

### How to Modify Training Configuration

**Method 1: Override specific parameters**
```bash
python training/train_attention_lstm.py -c config_attention_lstm \
    training.max_epochs=50 \
    model.hidden_size=128 \
    optimizer.lr=0.0005
```

**Method 2: Edit config file directly**
1. Open `config/config_attention_lstm.yaml`
2. Modify parameters:
   - `model.embedding_dim`: Embedding dimension
   - `model.hidden_size`: LSTM hidden size
   - `model.num_layers`: Number of LSTM layers
   - `model.bidirectional`: true/false
   - `training.max_epochs`: Maximum training epochs
   - `optimizer.lr`: Learning rate
   - `data.batch_size`: Batch size
   - `data.max_sequence_length`: Maximum sequence length
3. Save and train with: `python training/train_attention_lstm.py -c config_attention_lstm`

### Configuration Parameters Explained

**Data Configuration:**
- `data.size`: "full" or "small" (for testing)
- `data.max_sequence_length`: Maximum sequence length (sequences longer are truncated)
- `data.batch_size`: Batch size for training
- `data.train_split`: Train/validation split ratio (default: 0.8)

**Model Configuration:**
- `model.embedding_dim`: Size of action embeddings
- `model.hidden_size`: LSTM hidden state dimension
- `model.num_layers`: Number of stacked LSTM layers
- `model.bidirectional`: Use bidirectional LSTM (slower but better)
- `model.dropout`: Dropout rate for regularization
- `model.attention_dim`: Attention mechanism dimension

**Training Configuration:**
- `training.max_epochs`: Maximum number of training epochs
- `training.use_class_weights`: Balance imbalanced classes
- `training.use_focal_loss`: Use focal loss instead of cross-entropy
- `training.early_stopping.patience`: Early stopping patience

---

## How Training Works (Global Overview)

### Logistic Regression Pipeline

```
Raw Data → Clean Data → Feature Extraction → Train/Val Split → Scaling → Train Model → Evaluate → Save
```

1. **Data Loading**: Reads CSV files with variable columns
2. **Feature Extraction**: Extracts ~30 statistical features (counts, ratios, diversity metrics)
3. **Scaling**: StandardScaler normalizes features
4. **Training**: sklearn LogisticRegression with LBFGS solver
5. **Evaluation**: Accuracy, F1-score (macro/weighted), precision, recall

### Attention-LSTM Pipeline

```
Raw Data → Clean → Tokenize → Sequence Prep → Feature Fusion → Train LSTM+Attention → Evaluate → Save
```

1. **Data Preprocessing**:
   - Clean and parse action strings
   - Build vocabulary from unique action signatures
   - Encode sequences as integer IDs

2. **Feature Extraction**:
   - **Sequential features**: Action sequences (padded/truncated)
   - **Statistical features**: Handcrafted behavioral features
   - **Browser features**: One-hot encoded browser type

3. **Model Architecture**:
   - **Embedding Layer**: Maps action IDs to dense vectors
   - **LSTM Layers**: Process sequences to capture temporal patterns
   - **Attention Mechanism**: Weights important actions in sequences
   - **Fusion Layer**: Combines LSTM output + statistical features + browser features
   - **Classifier**: Multi-class classification over users

4. **Training Process**:
   - Uses PyTorch Lightning for training orchestration
   - Automatic checkpointing (saves best model based on validation F1)
   - Early stopping to prevent overfitting
   - Learning rate scheduling (ReduceLROnPlateau)
   - Class weights to handle imbalanced data

5. **Caching System**:
   - Stage 1: Tokenized sequences and vocabulary
   - Stage 2: Prepared RNN sequences (padded)
   - Stage 3: Statistical features
   - Stage 4: Cleaned dataframe
   - Speeds up subsequent training runs significantly

---

## Inference and Submission Generation

### Attention-LSTM Inference

Generate predictions on test data and create submission file:

```bash
python training/inference_attention_lstm.py \
    --checkpoint results/models/attention_lstm_lstm_YYYYMMDD_HHMMSS/best.ckpt \
    --output submission.csv
```

**Parameters:**
- `--checkpoint` or `-c`: Path to model checkpoint (.ckpt file)
- `--output` or `-o`: Output submission file path (default: `submission.csv`)
- `--test-path`: Test data path (default: `test`)
- `--data-dir`: Data directory (default: `data`)
- `--batch-size`: Batch size for inference (default: 64)
- `--train-path`: Training data path (for regenerating artifacts if missing)

**What it does:**
1. Loads trained model from checkpoint
2. Loads artifacts (vocabulary, user categories, scaler) from checkpoint directory
3. If artifacts missing, regenerates them from training data
4. Prepares test data (sequences + statistical features + browser features)
5. Generates predictions for all test samples
6. Maps predictions back to user IDs
7. Creates submission CSV file in the required format

**Example:**
```bash
# Using a specific checkpoint
python training/inference_attention_lstm.py \
    --checkpoint results/models/attention_lstm_lstm_20251031_165002/best.ckpt \
    --output my_submission.csv

# With custom paths
python training/inference_attention_lstm.py \
    --checkpoint results/models/attention_lstm_lstm_20251031_165002/best.ckpt \
    --test-path test \
    --data-dir data \
    --output submission.csv
```

**Output format:**
The submission file will have the format:
```csv
RowId,prediction
1,user_123
2,user_456
...
```

### Logistic Regression Inference

Inference is automatically performed during training. The model evaluates on both training and validation sets, and the results are saved with the model. To use the trained model for new predictions, load it using the `model_manager` utilities (see `model_manager.py` for API details).

---

## Results Storage

### Directory Structure

```
results/
├── models/                          # All trained models
│   ├── logistic_regression_*/      # Logistic Regression models
│   │   ├── *_model.joblib          # Trained model
│   │   ├── *_scaler.joblib         # Feature scaler
│   │   ├── *_metadata.json         # Model metadata
│   │   ├── *_features.joblib       # Feature names
│   │   └── *_categories.joblib     # User ID mappings
│   │
│   └── attention_lstm_*/           # Attention-LSTM models
│       ├── best.ckpt                # Best model checkpoint
│       ├── last.ckpt                # Last epoch checkpoint
│       ├── config.yaml              # Training configuration
│       └── artifacts/               # Required artifacts
│           ├── vocabulary.joblib   # Action vocabulary
│           ├── user_categories.joblib  # User ID mappings
│           └── stat_scaler.joblib  # Statistical feature scaler
│
└── metrics/                         # Training metrics
    ├── *_metrics.json               # Detailed metrics (train/val)
    └── ...
```

### TensorBoard Logs

TensorBoard logs are stored in `logs/` directory. To view training progress:

```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

