import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from utils import load_data, clean_data, prepare_training_data


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_topk_linear_svm_features(model, feature_names: pd.Index, k=15, out_dir="figures", filename="svm_topk_weights.png"):
    """Plot top-k absolute weights for linear SVM (after StandardScaler)."""
    _ensure_dir(out_dir)
    svc = model.named_steps["svc"]
    if not hasattr(svc, "coef_"):
        return
    # coef_ shape: (n_classes, n_features) one-vs-rest
    coef_abs = np.abs(svc.coef_)
    importances = coef_abs.mean(axis=0)
    idx = np.argsort(importances)[-k:][::-1]
    names = np.array(feature_names)[idx]
    vals = importances[idx]

    plt.figure(figsize=(7, 5))
    plt.barh(range(len(idx)), vals, alpha=0.85)
    plt.yticks(range(len(idx)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute mean weight")
    plt.title(f"SVM (linear) — Top {k} Feature Weights")
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("\nLoading training data")
    df_train = load_data("train")

    print("\nCleaning data")
    df_train_clean = clean_data(df_train, is_train=True)

    print("\nExtracting features")
    X, y, user_categories = prepare_training_data(df_train_clean)
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")

    print("\nTrain/validation split")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    print("\nTraining SVM (linear) with small C sweep")
    candidate_C = [0.1, 1, 10, 100]
    best_model = None
    best_val_f1 = -1.0
    best_C = None

    for C in candidate_C:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="linear",
                C=C,
                class_weight="balanced",
                probability=False,
                cache_size=2000,
                max_iter=100000
            ))
        ])
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        f1w = f1_score(y_val, pred_val, average="weighted")
        if f1w > best_val_f1:
            best_val_f1 = f1w
            best_C = C
            best_model = model

    print(f"Best C: {best_C} | Val F1-weighted: {best_val_f1:.4f}")

    print("\nEvaluation")
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)

    train_acc = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train, average="weighted")
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1 = f1_score(y_val, y_pred_val, average="weighted")

    print(f"Train: acc={train_acc:.4f} | f1w={train_f1:.4f}")
    print(f"Val: acc={val_acc:.4f} | f1w={val_f1:.4f}")

    # Visualizations
    print("\nVisualizations")
    plot_topk_linear_svm_features(best_model, X.columns, k=15, out_dir="figures", filename="svm_topk_weights.png")

    return best_model, X, y


if __name__ == "__main__":
    main()
