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

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    if depth >= max_depth:
        return True
    if len(labels) < min_samples_split:
        return True
    if impurity(labels) == 0.0:
        return True
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    # majority vote
    u, c = np.unique(labels, return_counts=True)
    combined = [(val, freq) for (val, freq) in zip(u, c)]
    combined.sort(key=lambda x: -x[1])
    return combined[0][0].item()

# Step 7 - build_tree
def build_tree(
    features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0
):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    # base case
    if should_stop(labels, depth, max_depth, min_samples_split):
        p = leaf_prediction(labels)
        return {"leaf": True, "prediction": p}
    # internal node
    result = {"leaf": False}
    # computing the best split
    if feature_subset is None:
        m = features.shape[1]
        feature_subset = list(range(m))
    if len(feature_subset) == 0:
        return {"leaf": True, "prediction": leaf_prediction(labels)}
    # computing the best split
    split_info = best_split(features, labels, feature_subset)
    if split_info["feature_index"] is None:
        return {"leaf": True, "prediction": leaf_prediction(labels)}
    feature_index = split_info["feature_index"]
    threshold = split_info["threshold"]

    # feature_subset = np.array(v for v in feature_subset if v != feature_index)
    feature_subset.remove(feature_index)
    l_features, l_labels, r_features, r_labels = split_dataset(
        features, labels, feature_index, threshold
    )

    left_node = build_tree(
        l_features, l_labels, max_depth, min_samples_split, feature_subset, depth + 1
    )
    right_node = build_tree(
        r_features, r_labels, max_depth, min_samples_split, feature_subset, depth + 1
    )

    return {
        "leaf": False,
        "feature_index": feature_index,
        "threshold": threshold,
        "left": left_node,
        "right": right_node,
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    # base case
    if tree['leaf']:
        return tree['prediction']
    else:
        feature_index, threshold = tree['feature_index'], tree['threshold']
        if example[feature_index] <= threshold:
            return predict_example_tree(tree['left'], example)
        else:
            return predict_example_tree(tree['right'], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    return np.array([predict_example_tree(tree, x) for x in features])

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    n = features.shape[0]
    idxs = rng.choice(range(n), size=n, replace=True)
    return features[idxs], labels[idxs]

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    return rng.choice(range(num_features), size = num_to_pick, replace=False)

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

