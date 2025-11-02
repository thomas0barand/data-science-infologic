import os
import numpy as np
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def align_test_columns(X_test: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
    """Align test matrix to training feature space: add missing columns (0), drop extras, and order columns."""
    missing = [c for c in train_columns if c not in X_test.columns]
    for c in missing:
        X_test[c] = 0
    extra = [c for c in X_test.columns if c not in train_columns]
    if extra:
        X_test = X_test.drop(columns=extra)
    return X_test[train_columns]

def save_submission(pred_codes: np.ndarray,
                    user_categories,
                    out_csv: str = "results/submissions/submission.csv"):
    """Map predicted integer codes to original user labels and save Kaggle-ready CSV."""
    ensure_dir(os.path.dirname(out_csv) or ".")

    def _codes_to_labels(codes: np.ndarray, cats_src):
        if hasattr(cats_src, "categories"):
            cats = cats_src.categories
            if isinstance(cats, (list, tuple)) and len(cats) > 0:
                first = cats[0]
                if isinstance(first, (list, tuple, pd.Index, np.ndarray)):
                    cats = first
            cats = np.array(cats)
            return np.array([cats[c] if 0 <= c < len(cats) else f"user_{c}" for c in codes])
        if isinstance(cats_src, (pd.Index, np.ndarray, list, tuple)):
            cats = np.array(cats_src)
            return np.array([cats[c] if 0 <= c < len(cats) else f"user_{c}" for c in codes])
        try:
            mapped = pd.Categorical.from_codes(codes, categories=cats_src)
            return np.asarray(mapped, dtype=str)
        except Exception:
            return np.array([str(c) for c in codes])

    user_ids = _codes_to_labels(pred_codes, user_categories)
    df_subm = pd.DataFrame({"prediction": user_ids})
    df_subm.index = np.arange(1, len(df_subm) + 1)
    df_subm.index.name = "RowId"
    df_subm.to_csv(out_csv, index=True)
    print(f"\nSaved submission: {out_csv}  (rows={len(df_subm)})")
