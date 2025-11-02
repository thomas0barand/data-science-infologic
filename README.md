# Data Science Infologic - Danylo Branch

## Description

This branch focuses on implementing and evaluating **tree-based machine learning models** for the Copilote user identification project.
Two main models are trained and compared:

1. **Random Forest** – robust ensemble method based on multiple decision trees.
2. **XGBoost** – gradient boosting framework optimized for high performance.

The goal is to classify users based on their behavioral traces recorded in the Copilote software.

---

## Project Structure

```
data-science-infologic/
├── data/                        # Dataset files
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── training/                    # Training scripts
│   ├── random_forest.py         # Random Forest training and evaluation
│   ├── train_xgboost.py         # XGBoost training and evaluation
│   ├── submission.py            # Common submission & column-alignment utilities
│   └── utils.py                 # Data loading and preprocessing
│
├── results/
│   ├── figures/                 # Visualizations (feature importances, statistics)
│   └── submissions/             # Kaggle submission files
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/thomas0barand/data-science-infologic.git
cd data-science-infologic
git checkout danylo_branch
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training Scripts

### XGBoost

```bash
python training/train_xgboost.py
```

**Workflow:**

* Loads and cleans data
* Extracts tabular features
* Trains an optimized XGBoost model
* Evaluates on validation split (80/20)
* Generates `results/submissions/submission.csv` for Kaggle

### Random Forest

```bash
python training/random_forest.py
```

**Workflow:**

* Same preprocessing pipeline
* Trains Random Forest with best hyperparameters
* Evaluates metrics and visualizes top features
* Exports predictions to `results/submissions/submission.csv`

---

## Results

**Feature visualizations** are stored in `results/figures/`:

* `rf_importances.png`
* `xgb_importances.png`
* other exploratory plots (browser usage, class balance, etc.)

**Submissions** (Kaggle format) are stored in `results/submissions/`:

* `submission.csv`
* `submission2.csv`
* `xgboost_submission.csv`
