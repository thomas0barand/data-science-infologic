"""
Training script for Copilote user identification using Linear Regression.
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils.utils import load_data, clean_data, prepare_training_data, prepare_test_data, ModelManager, save_model_results
# from model_manager import ModelManager, save_model_results


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
    
    # Scale features for better convergence
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = LogisticRegression(
        max_iter=5000,  # Increased iterations for convergence
        random_state=42,
        solver='lbfgs',
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train_scaled, y_train)
    print("   ✓ Model trained successfully")
    
    # Evaluate on validation set
    print("\n[6/6] Evaluating model on validation set...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    
    # Calculate comprehensive metrics using ModelManager
    manager = ModelManager()
    train_metrics = manager.calculate_comprehensive_metrics(y_train, y_pred_train)
    val_metrics = manager.calculate_comprehensive_metrics(y_val, y_pred_val)
    
    print(f"\n   Training Metrics:")
    print(f"   • Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"   • F1-Score (weighted): {train_metrics['f1_weighted']:.4f}")
    print(f"   • F1-Score (macro): {train_metrics['f1_score']:.4f}")
    
    print(f"\n   Validation Metrics:")
    print(f"   • Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"   • F1-Score (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"   • F1-Score (macro): {val_metrics['f1_score']:.4f}")
    
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
    
    # Save model and metrics using the new system
    print("\n[7/7] Saving model and metrics...")
    
    # Prepare additional metadata
    additional_metadata = {
        "num_features": X.shape[1],
        "num_classes": len(np.unique(y)),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "feature_names": list(X.columns),
        "model_params": {
            "max_iter": 5000,
            "solver": "lbfgs",
            "random_state": 42
        }
    }
    
    # Save using the new model management system
    model_path, metrics_path = save_model_results(
        model=model,
        model_type="logistic_regression",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        scaler=scaler,
        feature_columns=list(X.columns),
        user_categories=user_categories,
        additional_metadata=additional_metadata
    )
    
    print(f"   ✓ Model saved to: {model_path}")
    print(f"   ✓ Metrics saved to: {metrics_path}")
    
    # Create and display results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Get the model name from the path
    model_name = os.path.basename(os.path.dirname(model_path))
    summary = manager.get_model_summary(model_name)
    
    if summary["status"] == "complete":
        print(f"Model Name: {model_name}")
        print(f"Model Type: {summary['metadata']['model_type']}")
        print(f"Timestamp: {summary['metadata']['timestamp']}")
        print(f"Training Accuracy: {summary['metrics']['train_metrics']['accuracy']:.4f}")
        print(f"Validation Accuracy: {summary['metrics']['validation_metrics']['accuracy']:.4f}")
        print(f"Training F1 (weighted): {summary['metrics']['train_metrics']['f1_weighted']:.4f}")
        print(f"Validation F1 (weighted): {summary['metrics']['validation_metrics']['f1_weighted']:.4f}")
    
    print("=" * 80)
    
    return model, X, y


if __name__ == "__main__":
    main()

