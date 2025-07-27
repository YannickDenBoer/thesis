import pickle
import os
import numpy as np
import torch

CUE_LIST = ['Flexing', 'Finger splay', 'Eye squeeze', 'Hand-to-face', 'Hand-to-mouth', 
            'Mouth open', 'Adult hand', 'Yawn', 'Bottle feeding', 'Grasping', 
            'Hand sucking', 'Tongue out']

def load_data(path: str):
    # Load the embeddings and texts
    print(f"loading results from {path}...")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data

def save_data(data, path: str):
    # Save the embeddings and texts
    print(f"saving results to {path}...")
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save the embeddings and texts
    with open(path, "wb") as file:
        pickle.dump(data, file)

def sanitize_and_report(
    arr, 
    name, 
    group=None, 
    y_true=None, 
    label_names=None, 
    fill_value=0.0
):
    # move to numpy for easy masking
    is_tensor = isinstance(arr, torch.Tensor)
    a = arr.cpu().numpy() if is_tensor else arr

    n_samples = a.shape[0]
    # 1) find rows where *all* features are NaN
    #    flatten dims 1..end so it works for any shape beyond 2D
    all_nan_mask = np.isnan(a).reshape(n_samples, -1).all(axis=1)
    if all_nan_mask.any():
        drop_idxs = np.where(all_nan_mask)[0]
        print(f"Dropping {len(drop_idxs)} samples from {name} (all-NaN rows):")
        for i in drop_idxs:
            grp = group[i] if group is not None else None
            yt  = y_true[i] if y_true is not None else None
            print(f"  • sample={i}, group={grp}, y_true={yt}")
    # build mask of rows to *keep*
    keep_mask = ~all_nan_mask
    a_kept = a[keep_mask]
    group_kept = group[keep_mask] if group is not None else None
    y_true_kept = y_true[keep_mask] if y_true is not None else None

    # 2) within the kept rows, report any remaining NaN coords
    coords = np.argwhere(np.isnan(a_kept))
    if coords.size:
        print(f"Found {len(coords)} NaN entries in kept rows of {name}:")
        for idx in coords:
            i, *feat = idx    # sample idx in the *kept* array
            orig_i = np.nonzero(keep_mask)[0][i]  # map back to original index
            grp = group[orig_i] if group is not None else None
            # y_true for this feature (if 2D)
            if y_true is not None:
                yt = (y_true[orig_i, feat[0]] 
                      if y_true.ndim>1 else y_true[orig_i])
            else:
                yt = None
            lbl = (label_names[feat[0]] 
                   if label_names and feat 
                   else tuple(feat))
            print(f"  • sample={orig_i}, feat={tuple(feat)}, group={grp}, "
                  f"y_true={yt}, label={lbl}")

    # 3) now fill NaN / ±Inf in the kept array
    if is_tensor:
        cleaned = torch.nan_to_num(
            torch.from_numpy(a_kept), 
            nan=fill_value, 
            posinf=fill_value, 
            neginf=fill_value
        ).to(arr.dtype).to(arr.device)
    else:
        cleaned = np.nan_to_num(
            a_kept,
            nan=fill_value, 
            posinf=fill_value, 
            neginf=fill_value
        )
    
    return cleaned, y_true_kept, group_kept

def make_latex_table(df, caption, label,
                     float_fmt="%.3f",
                     multicolumn=True,
                     multicolumn_format='c',
                     escape=False):
    latex = df.to_latex(
        index=True,
        float_format=float_fmt,
        multicolumn=multicolumn,
        multicolumn_format=multicolumn_format,
        caption=caption,
        label=label,
        escape=escape
    )
    return latex