import numpy as np

def random_classifier(num_samples, num_labels, label_prob=0.5, random_state=42):
    rng = np.random.default_rng(random_state)
    y_random = rng.binomial(1, label_prob, size=(num_samples, num_labels))
    return y_random

def majority_classifier(y, num_samples, num_labels, k=1):
    y = np.array(y)
    y_majority = np.zeros((num_samples, num_labels), dtype=int)
    
    # Calculate the frequency (mean) of each label
    label_frequencies = np.mean(y, axis=0)
    
    # Get indices of top-k most frequent labels
    top_k_indices = np.argsort(-label_frequencies)[:k]
    
    # Assign 1 to the top-k labels for all samples
    y_majority[:, top_k_indices] = 1
    
    return y_majority

def weighted_classifier(y_train, num_samples, num_labels, random_state=42):
    y_train = np.array(y_train)
    y_weighted = np.zeros((num_samples, num_labels), dtype=int)

    # Set random seed if provided
    rng = np.random.default_rng(seed=random_state)
    
    # Calculate the frequency (mean) of each label in the training set
    label_frequencies = np.mean(y_train, axis=0)
    
    # For each sample, assign labels based on the frequency probabilities
    for i in range(num_samples):
        y_weighted[i] = rng.binomial(1, label_frequencies)
    
    return y_weighted