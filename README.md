# Data Science Infologic - User Identification Project

## Project Overview

This repository contains machine learning models for user identification based on interaction traces with the Copilote application. The project implements multiple classification approaches, each developed in separate branches by different contributors.

**Problem**: Given a sequence of user actions (button clicks, form entries, screen selections, etc.), identify which user performed those actions.

**Approach**: Multiple machine learning models ranging from traditional statistical methods to deep learning architectures.

**Repository Contents**: In this repository, you will find only the trained models, metrics, submissions, and data. Training scripts and development code are located in their respective branches.

---

## Repository Organization

This repository uses a **multi-branch architecture** where each branch contains a specific model implementation:

| Branch | Contributor | Models | Additional Contributions | Contents |
|--------|-------------|--------|-------------------------|----------|
| `main` | - | All models | - | Models, metrics, submissions, data |
| `thomas` | Thomas | Logistic Regression, LSTM + Attention | Preprocessing data (tokenization + statistical features) | Training scripts, code, configs |
| `dan` | Dan | XGBoost, Random Forest | Visualization and EDA | Training scripts, code, configs |
| `anna` | Anna | Decision Tree, SVM | EDA and documentation | Training scripts, code, configs |

**Important**: The `main` branch contains only trained models, metrics, submissions, and data. Each development branch (`thomas`, `dan`, `anna`) is self-contained with its own training scripts, configurations, and documentation. To train a specific model, you must switch to the corresponding branch.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/thomas0barand/data-science-infologic.git
cd data-science-infologic
```

### Switch to a Specific Branch

Each branch contains different model implementations. Choose the branch corresponding to the model you want to train:

```bash
# For Logistic Regression or LSTM + Attention
git checkout thomas

# For XGBoost or Random Forest
git checkout dan

# For Decision Tree or SVM
git checkout anna
```

### Installation

Once on your desired branch, follow the installation instructions in that branch's README:

1. **Using Poetry** (recommended):
   ```bash
   poetry install
   poetry shell
   ```

2. **Using requirements.txt**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Available Models

### Traditional Machine Learning Models

- **Logistic Regression** (`thomas` branch) - Statistical approach with handcrafted features
- **XGBoost** (`dan` branch) - Gradient boosting ensemble method
- **Random Forest** (`dan` branch) - Ensemble of decision trees
- **Decision Tree** (`anna` branch) - Tree-based classification
- **SVM** (`anna` branch) - Support Vector Machine classifier

### Deep Learning Models

- **LSTM + Attention** (`thomas` branch) - Sequence learning with attention mechanism

---

## Project Structure

Each branch follows a similar structure:

```
data-science-infologic/
├── config/              # Configuration files (YAML/Hydra)
├── training/            # Training scripts
│   ├── train_*.py      # Model-specific training scripts
│   ├── inference_*.py  # Inference scripts
│   ├── model/          # Model architectures
│   └── utils/          # Utilities (data loading, preprocessing)
├── data/               # Dataset files (train.csv, test.csv)
├── results/            # Training results (models, metrics)
├── cache/              # Preprocessed data cache
├── logs/               # TensorBoard logs
└── README.md           # Branch-specific documentation
```

---

## How to Train a Model

1. **Checkout the appropriate branch**:
   ```bash
   git checkout <branch-name>  # thomas, dan, or anna
   ```

2. **Install dependencies** (see branch README for details):
   ```bash
   poetry install  # or pip install -r requirements.txt
   ```

3. **Follow branch-specific instructions**:
   - Each branch has its own README with detailed training instructions
   - Training commands vary by model type
   - Configuration files are branch-specific

4. **Example** (for thomas branch):
   ```bash
   git checkout thomas
   poetry install
   poetry shell
   
   # Train Logistic Regression
   python training/train_logistic.py
   
   # Train LSTM + Attention
   python training/train_attention_lstm.py -c config_attention_lstm
   ```

---

## Contributing

This is a collaborative project. Each contributor works on their own branch:
- **Thomas**: `thomas` branch - Logistic Regression, LSTM + Attention, preprocessing data (tokenization + statistical features)
- **Dan**: `dan` branch - XGBoost, Random Forest, visualization and EDA
- **Anna**: `anna` branch - Decision Tree, SVM, EDA and documentation

To contribute or modify models:
1. Switch to the appropriate branch
2. Follow the branch-specific guidelines
3. Commit changes to that branch
4. Push to the remote repository

---

## Prerequisites

- Python >= 3.12, < 3.14
- Poetry or pip/virtualenv
- Git

Additional dependencies vary by branch. See each branch's README for specific requirements.
