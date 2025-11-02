import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from utils import load_data, clean_data, prepare_training_data
from submission import align_test_columns, save_submission

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_topk_xgb_importances(model: xgb.XGBClassifier,
                              feature_names: pd.Index,
                              k: int = 15,
                              out_dir: str = "results/figures",
                              filename: str = "xgb_importances.png"):
    """Save a horizontal bar chart with top-k XGBoost feature importances."""
    _ensure_dir(out_dir)
    if not hasattr(model, "feature_importances_"):
        print("No feature importances available.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[-k:][::-1]
    names = np.array(feature_names)[idx]
    vals = importances[idx]

    plt.figure(figsize=(7, 5))
    plt.barh(range(len(idx)), vals, alpha=0.85)
    plt.yticks(range(len(idx)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"XGBoost — Top {k} Feature Importances")
    plt.tight_layout()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"Saved figure: {save_path}")


def main():
    df_train = load_data("train")
    df_train_clean = clean_data(df_train, is_train=True)
    
    # # Filtering rare users 
    # MIN_SESSIONS = 10 
    # user_session_counts = df_train_clean['user_id'].value_counts() 
    # common_users = user_session_counts[user_session_counts >= MIN_SESSIONS].index 
    # df_train_filtered = df_train_clean[df_train_clean['user_id'].isin(common_users)] 
    # filtered_users = df_train_filtered['user_id'].nunique() 
    # print(f"Filtering complete. Using {filtered_users} users.")

    X, y, user_categories = prepare_training_data(df_train_clean)
    print(f"\nFeatures: {X.shape[1]}  Samples: {X.shape[0]}  Classes: {len(np.unique(y))}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nSplit:")
    print(f"train: {X_train.shape[0]},  val: {X_val.shape[0]}")

    num_classes = len(np.unique(y))
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=300
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    print(f"\nTrain: acc={accuracy_score(y_train, y_pred_train):.4f}  "
          f"f1_w={f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"Val: acc={accuracy_score(y_val, y_pred_val):.4f}  "
          f"f1_w={f1_score(y_val, y_pred_val, average='weighted'):.4f}")
    
    print("\nVisualisation of feature importances")
    plot_topk_xgb_importances(
        model, X.columns, k=15,
        out_dir="results/figures",
        filename="xgb_importances.png"
    )

    df_test = load_data("test")
    df_test_clean = clean_data(df_test, is_train=False)
    if "user_id" not in df_test_clean.columns:
        placeholder = user_categories[0] if len(user_categories) > 0 else "placeholder_user"
        df_test_clean = df_test_clean.copy()
        df_test_clean["user_id"] = placeholder

    try:
        X_test, _, _ = prepare_training_data(df_test_clean)
    except Exception:
        out = prepare_training_data(df_test_clean)
        X_test = out[0] if isinstance(out, (list, tuple)) else out

    X_test_aligned = align_test_columns(X_test.copy(), X.columns)
    preds_test_codes = model.predict(X_test_aligned)
    save_submission(preds_test_codes, user_categories, out_csv="results/submissions/submission.csv")

    return model, X, y


if __name__ == "__main__":
    main()
