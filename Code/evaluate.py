from utils import CUE_LIST
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    hamming_loss, classification_report, jaccard_score
)
import numpy as np

def evaluate(y_true, y_pred, label_names=CUE_LIST, zero_division=0, verbose=False):
    """
    Evaluate multilabel classification performance.
    
    Returns a dictionary with all metrics, which can be averaged across folds.
    """
    results = {}

    # Micro / Macro scores
    results["micro_precision"] = precision_score(y_true, y_pred, average="micro", zero_division=zero_division)
    results["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=zero_division)
    results["micro_recall"] = recall_score(y_true, y_pred, average="micro", zero_division=zero_division)
    results["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=zero_division)
    results["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=zero_division)
    results["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=zero_division)

    # Subset accuracy and hamming loss
    results["subset_accuracy"] = accuracy_score(y_true, y_pred)
    results["hamming_loss"] = hamming_loss(y_true, y_pred)

    # Jaccard Similarity
    results["jaccard_micro"] = jaccard_score(y_true, y_pred, average="micro")
    results["jaccard_macro"] = jaccard_score(y_true, y_pred, average="macro")

    # Label-based report (only if label names are given)
    if label_names is not None:
        report = classification_report(y_true, y_pred, target_names=label_names, zero_division=zero_division, output_dict=True)
        for label in label_names:
            for metric in ["precision", "recall", "f1-score"]:
                results[f"{label}_{metric}"] = report[label][metric]

    if verbose:
        for key, val in results.items():
            print(f"{key:20s}: {val:.4f}")

    return results

def summarize_labelwise(results_list, label_names):
    df = pd.DataFrame(results_list)
    # pick only label_* keys
    cols = [c for c in df.columns 
            if any(c.endswith(s) for s in ("_precision","_recall","_f1-score"))]

    # 1) compute mean/std and transpose so idx = "Label_metric", cols = ["mean","std"]
    agg = df[cols].agg(['mean','std']).T

    # 2) split index “Label_metric” → MultiIndex (label, metric)
    agg.index = pd.MultiIndex.from_tuples(
        [idx.rsplit('_',1) for idx in agg.index],
        names=['label','metric']
    )

    # 3) unstack metric → columns (statistic, metric)
    summary = agg.unstack(level='metric')

    # 4) swap so metric is outer, statistic inner
    summary = summary.swaplevel(0, 1, axis=1)
    summary.sort_index(axis=1, level=0, inplace=True)

    # 5) name your column‐levels
    summary.columns.names = ['metric','statistic']

    # 6) **re‐index rows and columns** to original orders
    # rows in the order of label_names
    summary = summary.reindex(label_names, axis=0)

    # columns in the order you want
    metric_order = ["precision", "recall", "f1-score"]
    summary = summary.reindex(metric_order, axis=1, level=0)
    # and ensure statistic order mean→std
    summary = summary.reindex(["mean","std"], axis=1, level=1)

    return summary
