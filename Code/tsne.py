import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import CUE_LIST
import os

ntoi = {'inf1': 1, 'inf2': 2, 'inf3':3 , 'inf4':4, 'inf5':5}

def plot_tsne(X, y, groups=None, perplexity=20, label_type='cue', label_names=CUE_LIST, cue_index=0, path=None):
    X = X.detach().cpu().numpy()
    print(f"X shape: {X.shape}")

    n_samples = X.shape[0]
    # at least 2, at most floor((n_samples-1)/3):
    max_perp = max(2, (n_samples - 1) // 3)
    perp    = min(perplexity, max_perp)

    tsne = TSNE(n_components=2, perplexity=perp, method='exact', random_state=42)
    X_tsne = tsne.fit_transform(X)

    colors = plt.cm.tab20(np.linspace(0, 1, len(label_names)))

    # Plot t-SNE result with colors for each cue
    plt.figure(figsize=(10, 8))

    if label_type == 'group':
        if groups is None:
            raise ValueError("groups must be provided when label_type='group'")
        unique_groups = np.unique(groups)
        sorted_groups = sorted(unique_groups, key=lambda g: ntoi.get(g, float('inf')))

        for group in sorted_groups:
            idx = groups == group
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f'Infant {ntoi[group]}', alpha=0.6)
        plt.legend(title='Videos')

    elif label_type == 'cue':
        # binary labels: 0=absent, 1=present
        label = y[:, cue_index].astype(int)
        cue_name = label_names[cue_index]

        # plot absent vs present
        idx_absent  = (label == 0)
        idx_present = (label == 1)

        plt.scatter(
            X_tsne[idx_absent, 0], X_tsne[idx_absent, 1],
            c='blue', label=f"Absent",
            alpha=0.6, s=30
        )
        plt.scatter(
            X_tsne[idx_present, 0], X_tsne[idx_present, 1],
            c='red', label=f"Present",
            alpha=0.6, s=30
        )
        plt.legend(title=f'Infant Cue: {cue_name}')

    elif label_type == 'time':
        time = np.arange(len(X))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=time, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Time Order')

    else:
        raise ValueError("label_type must be 'group', 'cue', or 'time'")

    #plt.title(f"t-SNE Visualization by {label_type.capitalize()}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    if path:
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'{label_type}.png')
        print(f"[t-SNE] saving plot to {save_path}")
        plt.savefig(save_path)
    #plt.show()