"""
Model and Metrics Management System

This module provides functions to handle model saving, metrics tracking,
and linking between models and their performance results.
"""

import json
import os
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


class ModelManager:
    """
    Manages model saving, metrics tracking, and result organization.
    """
    
    def __init__(self, base_dir: str = "results"):
        """
        Initialize the ModelManager.
        
        Args:
            base_dir (str): Base directory for storing results
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.metrics_dir = os.path.join(base_dir, "metrics")
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def generate_model_name(self, model_type: str, timestamp: Optional[str] = None) -> str:
        """
        Generate a logical model name with timestamp.
        
        Args:
            model_type (str): Type of model (e.g., 'logistic_regression', 'random_forest')
            timestamp (str, optional): Custom timestamp, defaults to current time
            
        Returns:
            str: Generated model name
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{model_type}_{timestamp}"
    
    def save_model_with_metadata(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        scaler: Optional[Any] = None,
        feature_columns: Optional[list] = None,
        user_categories: Optional[Any] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model with comprehensive metadata.
        
        Args:
            model: The trained model object
            model_name: Name for the model
            model_type: Type of the model
            scaler: Fitted scaler object
            feature_columns: List of feature column names
            user_categories: User category mappings
            additional_metadata: Any additional metadata to store
            
        Returns:
            str: Path to the saved model
        """
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save main model
        model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
        joblib.dump(model, model_path)
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = os.path.join(model_dir, f"{model_name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
        
        # Save feature columns if provided
        if feature_columns is not None:
            features_path = os.path.join(model_dir, f"{model_name}_features.joblib")
            joblib.dump(feature_columns, features_path)
        
        # Save user categories if provided
        if user_categories is not None:
            categories_path = os.path.join(model_dir, f"{model_name}_categories.joblib")
            joblib.dump(user_categories, categories_path)
        
        # Create and save metadata
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "scaler_path": scaler_path if scaler is not None else None,
            "features_path": features_path if feature_columns is not None else None,
            "categories_path": categories_path if user_categories is not None else None,
            "additional_metadata": additional_metadata or {}
        }
        
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def save_metrics(
        self,
        model_name: str,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save metrics results in JSON format.
        
        Args:
            model_name: Name of the model (should match saved model)
            train_metrics: Training set metrics
            val_metrics: Validation set metrics
            test_metrics: Test set metrics (optional)
            additional_info: Any additional information to store
            
        Returns:
            str: Path to the saved metrics file
        """
        metrics_data = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics or {},
            "additional_info": additional_info or {}
        }
        
        # Save metrics file
        metrics_filename = f"{model_name}_metrics.json"
        metrics_path = os.path.join(self.metrics_dir, metrics_filename)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        return metrics_path
    
    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics if probabilities are available
        if y_pred_proba is not None:
            # Log loss
            from sklearn.metrics import log_loss
            try:
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except:
                metrics['log_loss'] = None
        
        return metrics
    
    def load_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Load metadata for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        metadata_path = os.path.join(self.models_dir, model_name, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found for model: {model_name}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_metrics(self, model_name: str) -> Dict[str, Any]:
        """
        Load metrics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model metrics
        """
        metrics_path = os.path.join(self.metrics_dir, f"{model_name}_metrics.json")
        
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Metrics not found for model: {model_name}")
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def list_models(self) -> list:
        """
        List all available models.
        
        Returns:
            list: List of model names
        """
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, f"{item}_metadata.json")
                if os.path.exists(metadata_path):
                    models.append(item)
        
        return sorted(models)
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a model including metadata and metrics.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Complete model summary
        """
        try:
            metadata = self.load_model_metadata(model_name)
            metrics = self.load_metrics(model_name)
            
            return {
                "model_name": model_name,
                "metadata": metadata,
                "metrics": metrics,
                "status": "complete"
            }
        except FileNotFoundError as e:
            return {
                "model_name": model_name,
                "error": str(e),
                "status": "incomplete"
            }
    
    def create_results_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all models and their metrics.
        
        Returns:
            pd.DataFrame: Summary of all models
        """
        models = self.list_models()
        summary_data = []
        
        for model_name in models:
            try:
                summary = self.get_model_summary(model_name)
                if summary["status"] == "complete":
                    row = {
                        "model_name": model_name,
                        "model_type": summary["metadata"]["model_type"],
                        "timestamp": summary["metadata"]["timestamp"],
                        "train_accuracy": summary["metrics"]["train_metrics"].get("accuracy", None),
                        "val_accuracy": summary["metrics"]["validation_metrics"].get("accuracy", None),
                        "train_f1_weighted": summary["metrics"]["train_metrics"].get("f1_weighted", None),
                        "val_f1_weighted": summary["metrics"]["validation_metrics"].get("f1_weighted", None),
                    }
                    summary_data.append(row)
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
        
        return pd.DataFrame(summary_data)


def save_model_results(
    model: Any,
    model_type: str,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    scaler: Optional[Any] = None,
    feature_columns: Optional[list] = None,
    user_categories: Optional[Any] = None,
    test_metrics: Optional[Dict[str, float]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Convenience function to save model and metrics in one call.
    
    Args:
        model: The trained model object
        model_type: Type of the model
        train_metrics: Training set metrics
        val_metrics: Validation set metrics
        scaler: Fitted scaler object
        feature_columns: List of feature column names
        user_categories: User category mappings
        test_metrics: Test set metrics (optional)
        additional_metadata: Any additional metadata to store
        
    Returns:
        Tuple[str, str]: (model_path, metrics_path)
    """
    manager = ModelManager()
    
    # Generate model name
    model_name = manager.generate_model_name(model_type)
    
    # Save model
    model_path = manager.save_model_with_metadata(
        model=model,
        model_name=model_name,
        model_type=model_type,
        scaler=scaler,
        feature_columns=feature_columns,
        user_categories=user_categories,
        additional_metadata=additional_metadata
    )
    
    # Save metrics
    metrics_path = manager.save_metrics(
        model_name=model_name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        additional_info=additional_metadata
    )
    
    return model_path, metrics_path
