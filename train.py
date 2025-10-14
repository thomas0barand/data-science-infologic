"""
Training script for Copilote user identification using Linear Regression.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import load_data, clean_data, prepare_training_data, prepare_test_data


def main():
    print("=" * 80)
    print("Copilote User Identification - Linear Regression Training")
    print("=" * 80)
    
    # Load training data
    print("\n[1/6] Loading training data...")
    df_train = load_data("train")
    print(f"   ✓ Loaded {len(df_train)} training samples")
    print(f"   ✓ Data shape: {df_train.shape}")
    
    # Clean data
    print("\n[2/6] Cleaning data...")
    df_train_clean = clean_data(df_train, is_train=True)
    print(f"   ✓ Data cleaned")
    print(f"   ✓ Number of unique users: {df_train_clean['user_id'].nunique()}")
    
    # Prepare features and target
    print("\n[3/6] Extracting features...")
    X, y, user_categories = prepare_training_data(df_train_clean)
    print(f"   ✓ Features extracted: {X.shape[1]} features")
    print(f"   ✓ Feature names: {list(X.columns)}")
    print(f"   ✓ Target shape: {y.shape}")
    
    # Split data into train and validation sets
    print("\n[4/6] Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Training set: {X_train.shape[0]} samples")
    print(f"   ✓ Validation set: {X_val.shape[0]} samples")
    
    # Train Logistic Regression model (Linear model for classification)
    print("\n[5/6] Training Logistic Regression model...")
    print("   (Using Logistic Regression as it's a linear model suitable for classification)")
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")
    
    # Evaluate on validation set
    print("\n[6/6] Evaluating model on validation set...")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    val_f1 = f1_score(y_val, y_pred_val, average='weighted')
    
    print(f"\n   Training Metrics:")
    print(f"   • Accuracy: {train_accuracy:.4f}")
    print(f"   • F1-Score (weighted): {train_f1:.4f}")
    
    print(f"\n   Validation Metrics:")
    print(f"   • Accuracy: {val_accuracy:.4f}")
    print(f"   • F1-Score (weighted): {val_f1:.4f}")
    
    # Show top features (if available)
    if hasattr(model, 'coef_'):
        print(f"\n   Model has {model.coef_.shape[0]} classes and {model.coef_.shape[1]} features")
        # Calculate feature importance as mean absolute coefficient value across all classes
        feature_importance = np.abs(model.coef_).mean(axis=0)
        feature_names = X.columns
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        print("\n   Top 10 Most Important Features:")
        for idx in top_features_idx:
            print(f"   • {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    
    # Save model (optional)
    import joblib
    model_path = "linear_regression_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature columns for later use with test data
    feature_columns_path = "feature_columns.joblib"
    joblib.dump(list(X.columns), feature_columns_path)
    print(f"Feature columns saved to: {feature_columns_path}")
    
    # Save user categories for later prediction conversion
    user_categories_path = "user_categories.joblib"
    joblib.dump(user_categories, user_categories_path)
    print(f"User categories saved to: {user_categories_path}")
    
    return model, X, y


if __name__ == "__main__":
    main()

