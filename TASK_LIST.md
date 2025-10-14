# Task List - User Identification Project

## Project Goal
Predict which user is using the software based on their action traces.

**Team**: 3 people  
**Deadline**: November 2, 2025  
**Evaluation**: F1-Score

---

## Session 1: Data Exploration

### Person A - Data Loading
- [ ] Load train.csv and test.csv files
- [ ] Understand data structure (user_id, browser, actions)
- [ ] Count sessions per user

### Person B - Data Analysis  
- [ ] Analyze browser usage patterns
- [ ] Count different action types
- [ ] Look at session lengths

### Person C - Data Quality
- [ ] Check for missing values
- [ ] Find outliers
- [ ] Document data issues

---

## Session 2: Feature Engineering

### Person A - Basic Features
- [ ] Count total actions per session
- [ ] Extract browser information
- [ ] Create user frequency features

### Person B - Action Features
- [ ] Parse action types (buttons, screens, dialogs)
- [ ] Extract screen information from parentheses
- [ ] Count action frequencies

### Person C - Advanced Features
- [ ] Extract time patterns (t5, t10, t15 markers)
- [ ] Parse screen configurations (<DEFAUT>, etc.)
- [ ] Extract record chains ($AC$, $GP$)

---

## Session 3: Model Building

### Person A - Simple Models
- [ ] Logistic Regression
- [ ] Decision Tree
- [ ] Compare results

### Person B - Advanced Models
- [ ] Random Forest
- [ ] XGBoost
- [ ] Tune parameters

### Person C - Evaluation
- [ ] Calculate F1-scores
- [ ] Create confusion matrices
- [ ] Select best model

---

## Final Week: Submission

### All Team
- [ ] Apply best model to test data
- [ ] Format predictions for Kaggle
- [ ] Submit to competition
- [ ] Write 5-8 page report
- [ ] Submit report to Moodle

---

## Key Data Info
- **Train**: 3279 sessions, many columns
- **Test**: 324 sessions
- **Actions**: Variable length per session
- **Time**: t5, t10, t15 = 5-second intervals
- **Browsers**: Firefox, Chrome, Edge, Opera

## Deliverables
1. **Code**: Working Python scripts
2. **Report**: 5-8 page PDF
3. **Submission**: Kaggle predictions

**Deadline**: November 2, 2025
