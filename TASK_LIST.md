# Task List - User Identification Project (Copilote Software)

## Project Overview
**Objective**: Predict user identity from software usage traces using machine learning classification.

**Team**: 3 people  
**Deadline**: November 2, 2025  
**Evaluation Metric**: Average F1-Score

---

## Phase 1: Data Understanding & Exploration (Session 1)

### 1.1 Data Loading & Initial Analysis
- [ ] **Person A**: Load and examine the raw data structure
  - [ ] Implement `read_ds()` function for variable-column CSV files
  - [ ] Load train.csv (3279 rows, 14470 columns) and test.csv (324 rows, 7726 columns)
  - [ ] Understand the data format: user_id, browser, action sequences
  - [ ] Identify the maximum number of columns and padding with None values

- [ ] **Person B**: Explore data characteristics
  - [ ] Analyze browser distribution per user
  - [ ] Count sessions per user
  - [ ] Examine action types and their frequencies
  - [ ] Identify unique actions (excluding time markers like 't5', 't10', etc.)

- [ ] **Person C**: Data quality assessment
  - [ ] Check for missing values and outliers
  - [ ] Analyze session length distribution
  - [ ] Examine temporal patterns in actions
  - [ ] Document data quality issues

### 1.2 Data Structure Understanding
- [ ] **All Team**: Understand action format patterns
  - [ ] Screen creation: `Création d'un écran(infologic.core.accueil.Acc...)`
  - [ ] Dialog actions: `Affichage d'une dialogue`, `Fermeture d'une dialogue`
  - [ ] Button execution: `Exécution d'un bouton`
  - [ ] Time markers: `t5`, `t10`, `t15` (5-second intervals)
  - [ ] Screen configuration: `<DEFAUT>`, `<ACCUEIL_INST>`
  - [ ] Record identification: `$AC$`, `$GP$`
  - [ ] Modifications: actions ending with `1`

---

## Phase 2: Feature Engineering (Session 2)

### 2.1 Data Preprocessing
- [ ] **Person A**: Handle categorical variables
  - [ ] Convert user_id to categorical codes using `pd.Categorical`
  - [ ] Implement `to_categories()` function
  - [ ] Handle browser variable (4 values: Firefox, Google Chrome, Opera, Microsoft Edge)
  - [ ] Decide between categorical encoding vs One-Hot Encoding

- [ ] **Person B**: Action parsing and filtering
  - [ ] Implement `filter_action()` function to extract base actions
  - [ ] Parse screen information using regex: `pattern_ecran = re.compile(r"\((.*?)\)")`
  - [ ] Parse screen configuration: `pattern_conf_ecran = re.compile(r"<(.*?)>")`
  - [ ] Parse record chains: `pattern_chaine = re.compile(r"\$(.*?)\$")`

- [ ] **Person C**: Feature extraction
  - [ ] Count total actions per session
  - [ ] Extract most used screen per session
  - [ ] Extract most used screen configuration
  - [ ] Extract most used record chain
  - [ ] Create temporal features (session duration, action frequency)

### 2.2 Advanced Feature Engineering
- [ ] **All Team**: Create comprehensive feature set
  - [ ] Action frequency features (count of each action type)
  - [ ] Temporal features (time between actions, session duration)
  - [ ] Sequence features (action patterns, transitions)
  - [ ] Browser-specific features
  - [ ] Statistical features (mean, std, min, max of action counts)

---

## Phase 3: Model Development (Session 3)

### 3.1 Data Preparation for Modeling
- [ ] **Person A**: Train/Validation Split
  - [ ] Split data into training and validation sets
  - [ ] Implement cross-validation strategy
  - [ ] Handle class imbalance if present

- [ ] **Person B**: Feature Scaling and Selection
  - [ ] Apply appropriate scaling (StandardScaler, MinMaxScaler)
  - [ ] Feature selection techniques
  - [ ] Dimensionality reduction if needed

- [ ] **Person C**: Baseline Model
  - [ ] Implement simple baseline (e.g., most frequent class)
  - [ ] Calculate baseline F1-score
  - [ ] Document baseline performance

### 3.2 Model Implementation and Comparison
- [ ] **Person A**: Linear Models
  - [ ] Logistic Regression with regularization
  - [ ] Linear SVM
  - [ ] Tune hyperparameters

- [ ] **Person B**: Tree-based Models
  - [ ] Decision Tree
  - [ ] Random Forest
  - [ ] XGBoost
  - [ ] Tune hyperparameters (n_estimators, max_depth, etc.)

- [ ] **Person C**: Advanced Models
  - [ ] SVM with different kernels (polynomial, RBF)
  - [ ] Neural Networks (MLP)
  - [ ] Ensemble methods
  - [ ] Tune hyperparameters

### 3.3 Model Evaluation
- [ ] **All Team**: Comprehensive evaluation
  - [ ] Calculate F1-score (primary metric)
  - [ ] Calculate accuracy
  - [ ] Generate confusion matrices
  - [ ] Feature importance analysis
  - [ ] Cross-validation results

---

## Phase 4: Final Implementation & Submission

### 4.1 Best Model Selection
- [ ] **All Team**: Model comparison and selection
  - [ ] Compare all models on validation set
  - [ ] Select best performing model
  - [ ] Final hyperparameter tuning
  - [ ] Retrain on full training set

### 4.2 Test Set Processing
- [ ] **Person A**: Test data preprocessing
  - [ ] Apply same preprocessing pipeline to test.csv
  - [ ] Generate features for test set
  - [ ] Ensure feature consistency

- [ ] **Person B**: Prediction and formatting
  - [ ] Generate predictions on test set
  - [ ] Convert predictions back to user IDs
  - [ ] Format submission file according to sample_submission.csv

- [ ] **Person C**: Quality assurance
  - [ ] Validate submission format
  - [ ] Check prediction distribution
  - [ ] Final code review

### 4.3 Submission
- [ ] **All Team**: Final submission
  - [ ] Submit to appropriate Kaggle group:
    - TD-1 (8h-10h), Julien Velcin: https://www.kaggle.com/t/077eb1959d124b11971668f381767f06
    - TD-2 (8h-10h), Erwan Versmée: https://www.kaggle.com/t/2374f58cda734c69a28a8ba3ef6692e8
    - TD-3 (10h-12h), Julien Velcin: https://www.kaggle.com/t/ce3349734cc44028bad2875d462e7085
    - TD-4 (10h-12h), Erwan Versmée: https://www.kaggle.com/t/6435d33bed324430a9749337b04c2168

---

## Deliverables

### 1. Code (Re-executable)
- [ ] **All Team**: Clean, well-commented Python code
- [ ] Requirements.txt with all dependencies
- [ ] README.md with setup instructions
- [ ] Make code available on Git platform or provide archive

### 2. Report (5-8 pages PDF)
- [ ] **Person A**: Problem description and methodology
- [ ] **Person B**: Feature engineering and model development
- [ ] **Person C**: Results, analysis, and discussion
- [ ] **All Team**: Final review and submission to Moodle

### 3. Kaggle Submission
- [ ] Submit predictions to competition leaderboard
- [ ] Document final F1-score achieved

---

## Technical Notes

### Key Data Insights
- Train set: 3279 sessions, 14470 max columns
- Test set: 324 sessions, 7726 max columns
- Variable number of actions per session
- Time markers indicate 5-second intervals
- 4 browser types, multiple user IDs

### Important Patterns to Extract
- Screen usage patterns: `(infologic.core.gui.control...)`
- Configuration changes: `<DEFAUT>`, `<ACCUEIL_INST>`
- Record work: `$AC$`, `$GP$`
- Action modifications: actions ending with `1`
- Temporal behavior: time between actions

### Evaluation Focus
- **Primary metric**: Average F1-Score
- Balance precision and recall
- Consider class imbalance
- Use cross-validation for robust evaluation

---

## Timeline
- **Session 1**: Data exploration and understanding
- **Session 2**: Feature engineering and initial modeling
- **Session 3**: Model optimization and comparison
- **Final week**: Report writing and final submission

**Deadline**: November 2, 2025
