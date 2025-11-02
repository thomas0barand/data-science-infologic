import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from utils import load_data, clean_data, prepare_training_data


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_topk_feature_importances(model: DecisionTreeClassifier,
                                  feature_names: pd.Index,
                                  k: int = 15,
                                  out_dir: str = "figures",
                                  filename: str = "tree_feature_importances.png"):
    """Horizontal bar chart of top-k feature importances."""
    if not hasattr(model, "feature_importances_"):
        return
    _ensure_dir(out_dir)

    importances = model.feature_importances_
    idx = np.argsort(importances)[-k:][::-1]
    names = np.array(feature_names)[idx]
    vals = importances[idx]

    plt.figure(figsize=(7, 5))
    plt.barh(range(len(idx)), vals, alpha=0.8)
    plt.yticks(range(len(idx)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Decision Tree — Top {k} Feature Importances")
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    print("\nData loading")
    df_train = load_data("train")

    print("\nData cleaning")
    df_train_clean = clean_data(df_train, is_train=True)
    print(f"Users: {df_train_clean['user_id'].nunique()} | Rows: {len(df_train_clean)}")

    print("\nFilter rare users (min sessions = 10)")
    MIN_SESSIONS = 10
    counts = df_train_clean['user_id'].value_counts()
    keep_users = counts[counts >= MIN_SESSIONS].index
    df_train_filtered = df_train_clean[df_train_clean['user_id'].isin(keep_users)]
    print(f"Kept users: {df_train_filtered['user_id'].nunique()} | Rows: {len(df_train_filtered)}")

    print("\nFeatures")
    X, y, user_categories = prepare_training_data(df_train_filtered)
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")

    print("\nSplit")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    base = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [8, 12, 16, 20, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "min_samples_split": [2, 5, 10]
    }
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    model = DecisionTreeClassifier(random_state=42, class_weight='balanced', **best_params)
    model.fit(X_train, y_train)

    print("\nPost-pruning (cost-complexity)")
    try:
        path = model.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas
        if len(ccp_alphas) > 1:
            candidate_alphas = np.unique(
                np.linspace(ccp_alphas[0], ccp_alphas[-2], num=min(25, max(2, len(ccp_alphas)-1)))
            )
        else:
            candidate_alphas = np.array([0.0])

        best_alpha = 0.0
        best_val_f1 = -1.0
        for alpha in candidate_alphas:
            m = DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',
                criterion=best_params["criterion"],
                max_depth=best_params["max_depth"],
                min_samples_leaf=best_params["min_samples_leaf"],
                min_samples_split=best_params["min_samples_split"],
                ccp_alpha=alpha
            )
            m.fit(X_train, y_train)
            f1_tmp = f1_score(y_val, m.predict(X_val), average='weighted')
            if f1_tmp > best_val_f1:
                best_val_f1 = f1_tmp
                best_alpha = alpha

        model = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            criterion=best_params["criterion"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            min_samples_split=best_params["min_samples_split"],
            ccp_alpha=best_alpha
        )
        model.fit(X_train, y_train)
        print(f"Best ccp_alpha={best_alpha:.6f} | val f1w={best_val_f1:.4f}")
    except Exception as e:
        print(f"Skipped pruning: {e}")

    print("\nEvaluate")
    y_pred_tr = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_acc = accuracy_score(y_train, y_pred_tr)
    train_f1 = f1_score(y_train, y_pred_tr, average='weighted')
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1 = f1_score(y_val, y_pred_val, average='weighted')

    print(f"Train: acc={train_acc:.4f} | f1w={train_f1:.4f}")
    print(f"Val: acc={val_acc:.4f} | f1w={val_f1:.4f}")

    print("\nVisualizations")
    plot_topk_feature_importances(model, X.columns, k=15,
                                  out_dir="results/figures",
                                  filename="tree_feature_importances.png")

    return model, X, y


if __name__ == "__main__":
    main()
