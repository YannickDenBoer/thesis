import numpy as np
import pandas as pd

def check_label_distribution(y, splits, min_freq=2, rel_tol=0.34, label_names=None):
    results = []
    violations = []

    y = np.asarray(y)
    n_samples, n_labels = y.shape
    k_folds = len(splits)

    label_totals    = y.sum(axis=0)
    k = len(splits)
    ideal_val_counts   = label_totals / k
    ideal_train_counts = label_totals * (k - 1) / k

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        for split_name, idx, ideal in [
                ("train", train_idx, ideal_train_counts),
                ("val",   val_idx,   ideal_val_counts)
            ]:
            counts = y[idx].sum(axis=0)
            for i in range(n_labels):
                count = counts[i]
                rel_dev = abs(count - ideal[i]) / ideal[i]  if ideal[i] > 0 else 0
                ok_min = count >= min_freq
                ok_tol = rel_dev <= rel_tol

                if not ok_min:
                    violations.append(
                        f"Fold {fold_idx} [{split_name}] – {label_names[i]}: count {count} < min {min_freq}"
                    )
                if not ok_tol:
                    violations.append(
                        f"Fold {fold_idx} [{split_name}] – {label_names[i]}: deviation {rel_dev:.2f} > tol {rel_tol:.2f}"
                    )

                results.append({
                    "fold": fold_idx,
                    "split": split_name,
                    "label": label_names[i],
                    "count": count,
                    "ideal": ideal,
                    "rel_dev": rel_dev,
                    "ok_min_freq": ok_min,
                    "ok_rel_tol": ok_tol
                })

def calculate_grouping_metric(splits, groups):
    groups = np.array(groups)
    unique_groups = np.unique(groups)
    total_metric = 0
    total_count = 0

    for train_idx, val_idx in splits:
        train_groups = groups[train_idx]
        val_groups = groups[val_idx]

        train_counts = pd.Series(train_groups).value_counts()
        val_counts = pd.Series(val_groups).value_counts()

        all_groups = set(train_counts.index).union(val_counts.index)

        for g in all_groups:
            p_train = train_counts.get(g, 0)
            p_val = val_counts.get(g, 0)

            q_train = p_train / (p_train + p_val)
            q_val = p_val / (p_train + p_val)
            grouping_score = abs(q_train - q_val)
            total_metric += grouping_score
            total_count += 1

    return total_metric / total_count if total_count > 0 else 0