

from preprocessing import prepare_data
from cross_validation import grouped_iterative_multilabel_stratified_cross_validation
from utils import load_data, save_data, sanitize_and_report, CUE_LIST
from Models.model_tsne import Model, Processor
import numpy as np
from tsne import plot_tsne

if __name__ == "__main__":
    inference = True
    
    # Paths
    video_folder = ""
    annotation_file = ""
    results_folder = ""
    
    # Initialise model and processor/wrapper
    vl_model = Model()
    processor = Processor(vl_model)

    # Prepare Data
    folder = ""
    if inference:
        X, y, groups = prepare_data(video_folder, annotation_file)
        save_data((X,y,groups), folder)
    else:
        X, y, groups = load_data(folder)

    # Calculate Embedding voor all X
    if inference:
        X = processor.images_to_embeddings(X)
        save_data(X, results_folder + "embeddings.pkl")
    else:
        X = load_data(results_folder + "embeddings.pkl")

    # Clean NaN values
    X, y, groups = sanitize_and_report(X, "train_cos_scores", y_true=y, label_names=CUE_LIST, group = groups)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Plot T-SNE
    unique_groups = np.unique(groups)

    # Create image for a cue that clusters well on a single video vs a cue that does not cluster well on a single video
    for j in range(6):
        for i in range(len(CUE_LIST)):
            X_class = X[:, j, :]
            print(f"Class index: {i}")
            print("X_class shape:", X_class.shape)
            print("Has NaN:", np.isnan(X_class).any())
            print("Has Inf:", np.isinf(X_class).any())
            # Create image for each cue on open-question all vids
            plot_tsne(X_class, y, groups=groups, label_type='cue', label_names=CUE_LIST, cue_index=i, path=f"{results_folder}tsne/Open{j}/all/{CUE_LIST[i]}")
            # Create image for groups on closed-question all vids
            plot_tsne(X_class, y, groups=groups, label_type='group', label_names=CUE_LIST, cue_index=i, path=f"{results_folder}tsne/Open{j}/all/{CUE_LIST[i]}")
            plot_tsne(X_class, y, groups=groups, label_type='time', label_names=CUE_LIST, cue_index=i, path=f"{results_folder}tsne/Open{j}/all/{CUE_LIST[i]}")
            for group in unique_groups:
                # Single vids
                group_indices = np.where(groups == group)[0]
                X_group_class = X_class[group_indices]
                y_group_class = y[group_indices]
                plot_tsne(X_group_class, y_group_class, label_type='cue', label_names=CUE_LIST, cue_index=i, path= f"{results_folder}tsne/Open{j}/{group}/{CUE_LIST[i]}")
                plot_tsne(X_group_class, y_group_class, label_type='time', label_names=CUE_LIST, cue_index=i, path= f"{results_folder}tsne/Open{j}/{group}/{CUE_LIST[i]}")

