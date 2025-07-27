from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  
import numpy as np
from collections import defaultdict

def grouped_iterative_multilabel_stratified_cross_validation(
    X, y, groups, n_splits=5, epsilon=0, random_state=None
):
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_samples, n_labels = y.shape

    # Global label counts
    label_counts = y.sum(axis=0)
    # Remaining capacity per label per fold
    c_label = np.tile(label_counts / n_splits, (n_splits, 1)).astype(float)
    # Overall capacity per fold
    c_overall = c_label.sum(axis=1)

    # Group counts per fold
    G = defaultdict(lambda: np.zeros(n_splits, dtype=int))

    # Unassigned samples
    U = set(range(n_samples))

    # Labels sorted by increasing frequency
    labels_by_freq = np.argsort(label_counts)

    rng = np.random.RandomState(random_state)
    fold_assignments = np.full(n_samples, -1, dtype=int)

    for label in labels_by_freq:
        # Samples remaining with label ell
        D_ell = [i for i in U if y[i, label] == 1]
        rng.shuffle(D_ell)
        for i in D_ell:
            # Label-specific deficiency
            D_label = c_label[:, label]
            D_max = D_label.max()
            # Candidate folds within epsilon
            C0 = np.where(D_label >= D_max - epsilon)[0]

            # Soft group tie-break
            g = groups[i]
            group_counts = G[g][C0]
            max_group = group_counts.max()
            C1 = C0[group_counts == max_group]

            # Tie-break on overall capacity
            overall_c = c_overall[C1]
            max_overall = overall_c.max()
            C2 = C1[overall_c == max_overall]

            # Final random tie-break
            candidate = rng.choice(C2)

            # Assign sample to fold
            fold_assignments[i] = candidate
            # Update capacities
            labels_i = np.where(y[i] == 1)[0]
            for ell_prime in labels_i:
                c_label[candidate, ell_prime] -= 1
            c_overall[candidate] = c_label[candidate].sum()

            # Update group counts and unassigned set
            G[g][candidate] += 1
            U.remove(i)

    # Build train/validation splits
    splits = []
    for fold in range(n_splits):
        val_idx = np.where(fold_assignments == fold)[0]
        train_idx = np.where(fold_assignments != fold)[0]
        splits.append((train_idx, val_idx))

    return splits
