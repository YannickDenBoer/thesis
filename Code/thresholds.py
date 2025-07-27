import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

def find_local_thresholds(y_true, scores):
    n_classes = y_true.shape[1]
    thresholds = np.zeros(n_classes)
    f1s        = np.zeros(n_classes)
    
    for k in range(n_classes):
        p, r, thr = precision_recall_curve(y_true[:, k], scores[:, k])
        
        # safer F1: only divide where p + r > 0
        denom = p + r
        f1 = np.divide(2 * p * r, denom, out   = np.zeros_like(denom), where = denom > 0)
        
        best_idx = np.nanargmax(f1)
        thresholds[k] = thr[best_idx] if best_idx < len(thr) else 1.0
        f1s[k] = f1[best_idx]
    
    return thresholds, f1s

def find_global_threshold(y_true, scores):
    y_true_flat = y_true.ravel()
    scores_flat = scores.ravel()

    # compute precision, recall at all candidate thresholds
    p, r, thresholds = precision_recall_curve(y_true_flat, scores_flat)

    # vectorized F1 computation
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out   = np.zeros_like(denom), where = denom > 0)

    # pick the best
    best_idx = f1.argmax()
    best_t   = thresholds[best_idx]
    best_f1  = f1[best_idx]

    print(f"Global threshold = {best_t:.4f}, micro-F1 = {best_f1:.3f}")
    return best_t, best_f1

def convert_scores_to_labels(scores, threshold):
    scores = np.asarray(scores)
    if np.isscalar(threshold):
        return (scores >= threshold).astype(int)
    else:
        # broadcast per-class thresholds across rows
        thresh = np.asarray(threshold)
        return (scores >= thresh[None, :]).astype(int)