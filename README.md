# Data Science Infologic - Thomas Branch

## Description

This branch focuses on implementing and comparing two **classical machine learning algorithms** for user identification in the Copilote dataset:

1. **Support Vector Machine (SVM)** — a linear model trained with feature scaling and parameter sweep over C values.
2. **Decision Tree** — a tree-based model tuned via GridSearchCV and post-pruned using cost-complexity pruning.

Both models rely on the same feature engineering and preprocessing pipeline for fair comparison.

---

## Project Structure

```
data-science-infologic/
├── data/                         # Dataset files
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── training/                     # Model training scripts
│   ├── svm.py                    # Linear SVM training with scaling and C tuning
│   ├── decision_tree.py          # Decision Tree with hyperparameter search + pruning
│   └── utils.py                  # Shared utilities: data loading, cleaning, feature extraction
│
├── figures/                      # Model-specific figures
│   ├── svm_topk_weights.png
│   └── tree_feature_importances.png
│
├── figures/                      # Final exported figures
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/thomas0barand/data-science-infologic.git
cd data-science-infologic
git checkout thomas_branch
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training Scripts

### Support Vector Machine (SVM)

```bash
python training/svm.py
```

**What it does:**

* Loads and cleans dataset
* Extracts tabular behavioral features
* Splits data (80/20 train-validation)
* Trains Linear SVM with StandardScaler
* Performs parameter sweep over `C ∈ {0.1, 1, 10, 100}`
* Evaluates F1-weighted and accuracy
* Visualizes top weighted features (`svm_topk_weights.png`)

---

### Decision Tree

```bash
python training/decision_tree.py
```

**What it does:**

* Loads and cleans dataset
* Filters users with ≥10 sessions
* Performs grid search over tree parameters (criterion, depth, split/leaf size)
* Applies cost-complexity pruning
* Evaluates weighted F1-score and accuracy
* Saves top feature importances (`tree_feature_importances.png`)

---

**Author:** Anna Samsonenko
