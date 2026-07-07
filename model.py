"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # I will use GINI
    _, f = np.unique(labels, return_counts=True)
    n = len(labels)
    return 1.0 - ((f / n) ** 2).sum()

# Step 2 - split_dataset
def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    feature = features[:, feature_index]
    mask = feature <= threshold
    left_features = features[mask]
    left_labels = labels[mask]
    right_features = features[~mask]
    right_labels = labels[~mask]

    return (left_features, left_labels, right_features, right_labels)

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    parent_score = impurity(parent_labels)

    n = len(parent_labels)
    n1 = len(left_labels)
    n2 = len(right_labels)

    children_score = n1 / n * impurity(left_labels) + n2 / n * impurity(right_labels)

    return parent_score - children_score

# Step 4 - best_split
def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    feature_index = None
    threshold = None
    score = 0.0

    for i in feature_indices:
        # thirsholds are mid unique values
        feature = features[:, i]
        u = np.unique(feature)
        thresholds = (u[1:] + u[:-1]) / 2
        for t in thresholds:
            mask = feature <= t
            left_labels, right_labels = labels[mask], labels[~mask]
            splt_score = split_score(labels, left_labels, right_labels)
            if splt_score > score:
                score = splt_score
                feature_index = i
                threshold = t
    return {"feature_index": feature_index, "threshold": threshold, "score": score}

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

