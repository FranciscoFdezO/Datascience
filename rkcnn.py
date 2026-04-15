#!/usr/bin/env python3
"""
RkCNN: Random k Conditional Nearest Neighbor for classification.

This script implements the RkCNN algorithm from scratch (Lu and Gweon, 2025).
Load a CSV with features and a class column, then run classification with
progress printed to the terminal. Optionally save a 2D cluster plot (PCA or first two features).

Usage examples:
  python rkcnn.py data.csv
  python rkcnn.py data.csv --target class --k 3 --m 10 --r 50 --h 200
"""

# Import the module that reads command-line options (e.g. --target, --k).
import argparse
# Import the module that lets us exit the program and print to the terminal.
import sys
import os
# Import Counter so we can count how many points belong to each class.
from collections import Counter

# Import numpy for arrays and math (distances, random numbers, etc.).
import numpy as np

# Try to load pandas for reading CSV files; if not installed, we set it to None and print a message later.
try:
    import pandas as pd
except ImportError:
    pd = None
# Try to load matplotlib for drawing plots; if not installed, we set it to None and skip plots.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# --- Tuning constants: which values to try when the user runs with --tune ---
# k = how many nearest neighbors per class (smaller = more local; paper uses 1 or 3).
# m = how many features in each random subset (smaller can help when many features are noisy).
# r = how many of the best subsets we keep (more = often better accuracy).
# h = how many random subsets we sample (more = bigger pool to pick the best from).

# Lists below are the values we try when tuning. None means "use a default value" for that setting.
TUNE_K = [1, 3, 5]
TUNE_M = None   # If None, we use one default (square root of number of features).
TUNE_R = [100, 200, 300]
TUNE_H = None   # If None, we set h = 3*r for each r we try.


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def euclidean_distance(x, X):
    """
    Compute Euclidean distance from a single point x to each row of X.
    x: 1D array of shape (m,)
    X: 2D array of shape (n, m)
    Returns: 1D array of shape (n,) with distances.
    """
    # Subtract the point from each row to get the difference along each feature.
    diff = X - x
    # Square the differences, add them up along each row, then take the square root to get distance.
    return np.sqrt(np.sum(diff * diff, axis=1))


def kth_neighbor_distance_per_class(X_sub, y_train, x_query, k, classes):
    """
    For each class, find the distance from x_query to the k-th nearest training
    point in that class (in the subspace X_sub). If a class has fewer than k
    points, use the farthest available neighbor for that class.

    X_sub: 2D array (n_train, n_features_sub), training data in one feature subset.
    y_train: 1D array of class labels for each training row.
    x_query: 1D array, query point in same subspace.
    k: number of nearest neighbors to consider per class.
    classes: list or array of unique class labels (order preserved for output).

    Returns: 1D array of length len(classes): distance to k-th (or last) neighbor in each class.
    """
    # Make sure the query point is a flat array of numbers.
    x_query = np.asarray(x_query, dtype=float).ravel()
    # Get the distance from the query point to every training point (in this feature subset).
    distances = euclidean_distance(x_query, X_sub)
    # We will fill in one distance per class.
    out = np.zeros(len(classes))
    for idx, c in enumerate(classes):
        # Find which training rows belong to this class.
        mask = y_train == c
        if not np.any(mask):
            # No points in this class; set distance to infinity so we never pick it.
            out[idx] = np.inf
            continue
        # Get distances only for points in this class.
        d_c = distances[mask]
        # Sort so the nearest is first and the k-th nearest is at index k-1.
        d_c = np.sort(d_c)
        # If the class has fewer than k points, use the last (farthest) one.
        pos = min(k - 1, len(d_c) - 1)
        out[idx] = d_c[pos]
    return out


def separation_score(X_sub, y):
    """
    Compute the separation score S = BV / WV for one feature subset.

    Per paper: BV uses L2 norms (not squared); WV uses L2 norms of sample deviations.
    BV = (1/(L-1)) * sum over classes of ||mean_c - overall_mean|| (L2 norm).
    WV = (1/L) * sum over classes of [ (1/(Nc-1)) * sum over points of ||point - mean_c|| ] (L2 norms).

    X_sub: 2D array (n_samples, n_features_sub).
    y: 1D array of class labels.

    Returns: float S, or 0.0 if computation is invalid (e.g. one class, or Nc=1 for some class).
    """
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    L = len(classes)
    if L <= 1:
        return 0.0
    overall_mean = np.mean(X_sub, axis=0)
    BV_sum = 0.0
    WV_sum = 0.0
    for c in classes:
        mask = y == c
        X_c = X_sub[mask]
        Nc = X_c.shape[0]
        if Nc <= 1:
            return 0.0
        mean_c = np.mean(X_c, axis=0)
        BV_sum += np.sqrt(np.sum((mean_c - overall_mean) ** 2))
        wv_c = np.sum(np.sqrt(np.sum((X_c - mean_c) ** 2, axis=1))) / (Nc - 1)
        WV_sum += wv_c
    BV = BV_sum / (L - 1)
    WV = WV_sum / L
    if WV <= 0:
        return 0.0
    return BV / WV


def separation_scores_per_feature(X, y):
    """
    Compute separation score (BV/WV) for each single feature.
    Returns 1D array of length X.shape[1]. Higher score = better class separation.
    """
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    # For each feature column, compute how well it separates the classes.
    for j in range(n_features):
        scores[j] = separation_score(X[:, j : j + 1], y)
    return scores


def kcnn_predict_one(X_train_sub, y_train, x_query, k, classes):
    """
    Get class probabilities for a single query using kCNN on one feature subset.

    Probability for class c is proportional to 1/d_c where d_c is the distance
    to the k-th nearest neighbor in class c. Then normalize to sum to 1.

    If all distances are zero or infinite, return uniform probabilities.

    Returns: 1D array of length len(classes), probabilities in same order as classes.
    """
    # Get the distance to the k-th nearest neighbor in each class for this query point.
    d_per_class = kth_neighbor_distance_per_class(X_train_sub, y_train, x_query, k, classes)
    # Avoid dividing by zero: replace zero or negative distances with a tiny number.
    d_safe = np.where(d_per_class <= 0, 1e-10, d_per_class)
    # Closer neighbor means higher score; we use 1/sqrt(distance) so smaller distance = larger value (softer than 1/d).
    inv_d = 1.0 / np.sqrt(d_safe)
    inv_d[np.isinf(d_per_class)] = 0.0
    total = np.sum(inv_d)
    if total <= 0:
        # If no valid distances, give each class the same probability.
        return np.ones(len(classes)) / len(classes)
    # Scale so the probabilities add up to 1.
    return inv_d / total


# ---------------------------------------------------------------------------
# RkCNN fit and predict
# ---------------------------------------------------------------------------

def rkcnn_fit(X, y, k, m, r, h, random_state=None, verbose=True, use_class_weights=False):
    """
    Fit RkCNN: sample h random feature subsets of size m, compute separation
    scores, keep top r subsets and their weights. No kCNN models are stored;
    we only store subset indices and weights. Prediction will use X and y from
    this fit (passed to predict).

    X: 2D array (n_samples, n_features).
    y: 1D array of class labels.
    k: number of nearest neighbors per class in kCNN.
    m: number of features in each random subset.
    r: number of top subsets to use for prediction.
    h: total number of random subsets to sample.
    random_state: for reproducibility.
    verbose: if True, print progress to terminal.
    use_class_weights: if True, store class weights (n / (n_classes * count_c)) for balanced prediction.

    Returns: dict with keys:
      top_subset_indices: list of r arrays, each array is the feature indices for that subset.
      weights: 1D array of length r (sum to 1).
      classes: unique class labels in order used for probabilities.
      X_train, y_train: stored for use in predict.
      class_weights: (optional) 1D array, same order as classes, for use in predict when use_class_weights.
    """
    # Set up the random number generator so we can repeat the same run later.
    rng = np.random.default_rng(random_state)
    n_samples, q = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # Do not use more features than we have, or more top subsets than we sampled.
    if m > q:
        m = q
    if r > h:
        r = h

    # Build a list of h random subsets; each subset has m different feature indices.
    all_subsets = []
    for j in range(h):
        if verbose:
            print(f"  Sampling random feature subset {j + 1} of {h}")
        idx = rng.choice(q, size=m, replace=False)
        all_subsets.append(np.sort(idx))

    if verbose:
        print(f"  Computing separation scores for {h} subsets.")

    # For each subset, compute how well it separates the classes (higher score = better).
    scores = np.zeros(h)
    for j in range(h):
        X_sub = X[:, all_subsets[j]]
        scores[j] = separation_score(X_sub, y)

    # Sort subsets by score from best to worst and keep only the top r.
    order = np.argsort(-scores)
    top_r_order = order[:r]
    top_scores = scores[top_r_order]
    top_subset_indices = [all_subsets[j] for j in top_r_order]

    # Turn the top scores into weights that add up to 1 (better score = higher weight).
    weight_sum = np.sum(top_scores)
    if weight_sum <= 0:
        weights = np.ones(r) / r
    else:
        weights = top_scores / weight_sum

    if verbose:
        print(f"  Using top {r} subsets. Building kCNN 1 of {r} ... {r} of {r} (models built at prediction time).")

    # Optional: class weights for imbalanced data (minority class gets higher weight at prediction time).
    out = {
        "top_subset_indices": top_subset_indices,
        "weights": weights,
        "classes": classes,
        "X_train": X,
        "y_train": y,
        "k": k,
    }
    if use_class_weights:
        class_weights = np.zeros(n_classes)
        for i, c in enumerate(classes):
            count_c = np.sum(y == c)
            if count_c > 0:
                class_weights[i] = n_samples / (n_classes * count_c)
            else:
                class_weights[i] = 1.0
        out["class_weights"] = class_weights
    return out


def rkcnn_predict(fit_result, X_new, verbose=True):
    """
    Predict class labels and optionally class probabilities for X_new using
    the fitted RkCNN (result of rkcnn_fit).

    Progress: print every 10% of samples so the user sees the algorithm is running.
    """
    # Get the stored subsets, weights, class list, and training data from the fit step.
    top_subset_indices = fit_result["top_subset_indices"]
    weights = fit_result["weights"]
    classes = fit_result["classes"]
    X_train = fit_result["X_train"]
    y_train = fit_result["y_train"]
    k = fit_result["k"]

    # Make sure the new data is a 2D array of numbers; if one row was passed, make it 2D.
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    n_new = X_new.shape[0]
    r = len(top_subset_indices)

    # We will fill in one probability per class for each new point.
    proba = np.zeros((n_new, len(classes)))
    for i in range(n_new):
        if verbose:
            if i == 0:
                print(f"  Predicting sample 1 of {n_new}")
            elif (i + 1) % max(1, n_new // 10) == 0 or i == n_new - 1:
                print(f"  Predicting sample {i + 1} of {n_new}")
        # Combine votes from all top subsets: each subset gives probabilities, we weight and add them.
        p_combined = np.zeros(len(classes))
        for j in range(r):
            idx = top_subset_indices[j]
            X_train_sub = X_train[:, idx]
            x_query = X_new[i, idx]
            p_j = kcnn_predict_one(X_train_sub, y_train, x_query, k, classes)
            p_combined += weights[j] * p_j
        proba[i] = p_combined

    # Optionally apply class weights (minority class gets higher effective weight) for the decision only.
    if "class_weights" in fit_result:
        scores = proba * fit_result["class_weights"]
    else:
        scores = proba

    # For each point, pick the class that has the highest (weighted) score.
    pred_labels = classes[np.argmax(scores, axis=1)]
    return pred_labels, proba


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv_for_classification(csv_path, target_column=None, target_index=None):
    """
    Load a CSV file for classification. One column is the target (class);
    all other numeric columns are features. Rows with missing target or
    non-numeric feature values are dropped.

    target_column: name of the column containing the class label (optional).
    target_index: 0-based index of the target column if target_column not given (optional).
    If neither is given, the last column is used as target.

    Returns: (X, y, feature_names, target_name)
    """
    if pd is None:
        raise ImportError("pandas is required for CSV loading. Install with: pip install pandas")

    df = pd.read_csv(csv_path)
    # Decide which column is the class (target) and which columns are features.
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found. Columns: {list(df.columns)}")
        target_name = target_column
        y = df[target_column]
        feature_cols = [c for c in df.columns if c != target_column]
    elif target_index is not None:
        target_name = df.columns[target_index]
        y = df.iloc[:, target_index]
        feature_cols = [c for i, c in enumerate(df.columns) if i != target_index]
    else:
        # If neither given, use the last column as the class.
        target_name = df.columns[-1]
        y = df.iloc[:, -1]
        feature_cols = list(df.columns[:-1])

    X_df = df[feature_cols]
    # Turn feature columns into numbers; non-numeric cells become NaN.
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    # Keep only rows where the target and all features are valid numbers.
    valid = ~(y.isna() | X_df.isna().any(axis=1))
    X = X_df.loc[valid].values
    y = y.loc[valid].values
    # Convert class labels to strings so we can use any label names.
    y = np.array([str(v) for v in y])

    return X, y, feature_cols, target_name


# ---------------------------------------------------------------------------
# Visualization: cluster plots by top features (separation score, no PCA)
# ---------------------------------------------------------------------------

def plot_separation_scores(scores, feature_names=None, filepath="separation_scores.png", max_features=None):
    """
    Bar chart of separation score per feature, sorted descending (highest first).
    Shows which feature has the highest vs lowest separation score.
    If max_features is set, only the top max_features are shown (e.g. to avoid huge plots).
    """
    if plt is None:
        print("  matplotlib not available; skipping separation scores plot.")
        return
    scores = np.asarray(scores)
    n = len(scores)
    # Sort so the best-separating feature is first.
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels = [str(feature_names[order[i]]) if feature_names is not None else str(order[i]) for i in range(n)]
    # If max_features is set, show only the top that many (so we don't draw 500 bars).
    if max_features is not None and n > max_features:
        scores_sorted = scores_sorted[:max_features]
        labels = labels[:max_features]
        n = len(scores_sorted)
    fig, ax = plt.subplots(1, 1, figsize=(max(8, n * 0.3), 5))
    x_pos = np.arange(n)
    ax.bar(x_pos, scores_sorted, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Separation score")
    ax.set_xlabel("Feature")
    title = "Separation score by feature (highest to lowest)"
    if max_features is not None and len(scores) > max_features:
        title = "Separation score by feature (top {} of {})".format(max_features, len(scores))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {filepath}")


def plot_clusters_2d(X, y_true=None, y_pred=None, filepath="clusters_2d.png", title=None, feature_names=None, top_two_indices=None):
    """
    Plot data in 2D with two panels: left = true class, right = predicted class (RkCNN).
    Uses top 2 features by separation score (no PCA). Saves figure to filepath.
    """
    if plt is None:
        print("  matplotlib not available; skipping 2D cluster plot.")
        return
    n_features = X.shape[1]
    if n_features == 0:
        return
    labels_for_ranking = y_true if y_true is not None else y_pred
    # If not given, pick the two features that separate classes best.
    if top_two_indices is None and labels_for_ranking is not None:
        scores = separation_scores_per_feature(X, labels_for_ranking)
        top_two_indices = np.argsort(-scores)[: min(2, n_features)]
    elif top_two_indices is None:
        top_two_indices = np.array([0, min(1, n_features - 1)])
    if len(top_two_indices) < 2:
        top_two_indices = np.array([top_two_indices[0], top_two_indices[0]])
    i, j = top_two_indices[0], top_two_indices[1]
    X_2d = np.column_stack([X[:, i], X[:, j] if j != i else np.zeros(X.shape[0])])
    xlab = str(feature_names[i]) if feature_names is not None else "Feature {}".format(i)
    ylab = str(feature_names[j]) if feature_names is not None else "Feature {}".format(j)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left panel: points colored by true class. Right panel: points colored by predicted class.
    for ax, labels, panel_title in [
        (axes[0], y_true, "True class"),
        (axes[1], y_pred, "Predicted class (RkCNN)"),
    ]:
        if labels is None:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
            ax.set_title(panel_title)
        else:
            uniq = np.unique(labels)
            for lab in uniq:
                mask = np.array(labels) == lab
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=str(lab), alpha=0.7)
            ax.legend()
            ax.set_title(panel_title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {filepath}")


def plot_clusters_3d(X, y_true=None, y_pred=None, filepath="clusters_3d.png", title=None, feature_names=None, top_three_indices=None):
    """
    Plot data in 3D with two panels: left = true class, right = predicted class (RkCNN).
    Uses top 3 features by separation score (no PCA). Saves figure to filepath.
    """
    if plt is None:
        print("  matplotlib not available; skipping 3D cluster plot.")
        return
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("  mplot3d not available; skipping 3D cluster plot.")
        return
    n_features = X.shape[1]
    if n_features == 0:
        return
    labels_for_ranking = y_true if y_true is not None else y_pred
    # If not given, pick the three features that separate classes best.
    if top_three_indices is None and labels_for_ranking is not None:
        scores = separation_scores_per_feature(X, labels_for_ranking)
        top_three_indices = np.argsort(-scores)[: min(3, n_features)]
    elif top_three_indices is None:
        top_three_indices = np.array([0, min(1, n_features - 1), min(2, n_features - 1)])
    idx_arr = np.array(top_three_indices)
    if len(idx_arr) < 3:
        idx_arr = np.resize(idx_arr, 3)
    i, j, k = idx_arr[0], idx_arr[1], idx_arr[2]
    # Build a 3D view: one column per chosen feature.
    X_3d = np.zeros((X.shape[0], 3))
    X_3d[:, 0] = X[:, i]
    X_3d[:, 1] = X[:, j] if j < n_features else np.zeros(X.shape[0])
    X_3d[:, 2] = X[:, k] if k < n_features else np.zeros(X.shape[0])
    dim_label = (
        str(feature_names[i]) if feature_names is not None else "Feature {}".format(i),
        str(feature_names[j]) if feature_names is not None else "Feature {}".format(j),
        str(feature_names[k]) if feature_names is not None else "Feature {}".format(k),
    )
    fig = plt.figure(figsize=(14, 6))
    # Left panel: true class. Right panel: predicted class.
    for p, (labels, panel_title) in enumerate([
        (y_true, "True class"),
        (y_pred, "Predicted class (RkCNN)"),
    ]):
        ax = fig.add_subplot(1, 2, p + 1, projection="3d")
        if labels is None:
            ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], alpha=0.7)
            ax.set_title(panel_title)
        else:
            uniq = np.unique(labels)
            for lab in uniq:
                mask = np.array(labels) == lab
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], label=str(lab), alpha=0.7)
            ax.legend()
            ax.set_title(panel_title)
        ax.set_xlabel(dim_label[0])
        ax.set_ylabel(dim_label[1])
        ax.set_zlabel(dim_label[2])
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {filepath}")


def plot_clusters_top_features_pairwise(X, y_true=None, y_pred=None, top_indices=None, feature_names=None, filepath="clusters_top5_features.png", n_top=5):
    """
    Pairwise scatter plots for the top n_top features by separation score.
    One subplot per pair (i, j) with i < j; points colored by predicted class.
    """
    if plt is None:
        print("  matplotlib not available; skipping pairwise top-features plot.")
        return
    n_features = X.shape[1]
    if n_features < 2:
        return
    labels_for_ranking = y_true if y_true is not None else y_pred
    # If not given, pick the top features by separation score.
    if top_indices is None and labels_for_ranking is not None:
        scores = separation_scores_per_feature(X, labels_for_ranking)
        top_indices = np.argsort(-scores)[: min(n_top, n_features)]
    elif top_indices is None:
        top_indices = np.arange(min(n_top, n_features))
    top_indices = np.asarray(top_indices)[: min(n_top, n_features)]
    n_top_actual = len(top_indices)
    n_pairs = n_top_actual * (n_top_actual - 1) // 2
    if n_pairs == 0:
        return
    n_cols = min(5, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_pairs == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    pair_idx = 0
    # For each pair of top features, draw a scatter plot (x = one feature, y = other).
    for ii in range(n_top_actual):
        for jj in range(ii + 1, n_top_actual):
            i, j = top_indices[ii], top_indices[jj]
            row, col = pair_idx // n_cols, pair_idx % n_cols
            ax = axes[row, col]
            labels = y_pred if y_pred is not None else y_true
            if labels is None:
                ax.scatter(X[:, i], X[:, j], alpha=0.7)
            else:
                uniq = np.unique(labels)
                for lab in uniq:
                    mask = np.array(labels) == lab
                    ax.scatter(X[mask, i], X[mask, j], label=str(lab), alpha=0.7)
                ax.legend(fontsize=8)
            xlab = str(feature_names[i]) if feature_names is not None else "f{}".format(i)
            ylab = str(feature_names[j]) if feature_names is not None else "f{}".format(j)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title("{} vs {}".format(xlab, ylab), fontsize=9)
            pair_idx += 1
    # Hide any extra subplot slots we did not use.
    for idx in range(pair_idx, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    fig.suptitle("Pairwise scatter: top {} features by separation (predicted class)".format(n_top_actual), y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {filepath}")


def plot_per_class_accuracy(y_true, y_pred, classes, filepath="accuracy_by_class.png", title=None):
    """
    Bar chart of percentage of points correctly predicted for each class.
    For each class c: (number correct in class c) / (total true in class c) * 100.
    Saves figure to filepath and prints path.
    """
    if plt is None:
        print("  matplotlib not available; skipping per-class accuracy plot.")
        return

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    n_classes = len(classes)
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    # For each class, count how many points are in that class and how many we got right.
    for i, c in enumerate(classes):
        mask = y_true == c
        total[i] = np.sum(mask)
        correct[i] = np.sum((y_true == c) & (y_pred == c))
    # Percentage correct per class; avoid dividing by zero if a class has no points.
    pct = np.where(total > 0, 100.0 * correct / total, 0.0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x_pos = np.arange(n_classes)
    bars = ax.bar(x_pos, pct, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_ylabel("Percentage correct")
    ax.set_xlabel("Class")
    ax.set_ylim(0, 105)
    ax.set_title(title or "Accuracy by class (percentage of points correct)")
    # Write the percentage on top of each bar.
    for i, (bar, val) in enumerate(zip(bars, pct)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, "{:.0f}%".format(val), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {filepath}")


# ---------------------------------------------------------------------------
# Metrics and main
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    # Fraction of points where the predicted label matches the true label.
    return np.mean(np.array(y_true) == np.array(y_pred))


def confusion_matrix(y_true, y_pred, classes):
    # Build a table: row i = true class, column j = predicted class; entry = count.
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            cm[i, j] = np.sum((y_true == c) & (y_pred == d))
    return cm


def balanced_accuracy(y_true, y_pred, classes):
    # Mean of per-class recall (sensitivity). Good for imbalanced data; 0.5 = no better than random for binary.
    cm = confusion_matrix(y_true, y_pred, classes)
    n = len(classes)
    recalls = []
    for i in range(n):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            recalls.append(cm[i, i] / row_sum)
        else:
            recalls.append(0.0)
    return np.mean(recalls) if recalls else 0.0


def min_per_class_recall(y_true, y_pred, classes):
    # Minimum recall across classes; 0 if one class has 0% recall.
    cm = confusion_matrix(y_true, y_pred, classes)
    n = len(classes)
    recalls = []
    for i in range(n):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            recalls.append(cm[i, i] / row_sum)
        else:
            recalls.append(0.0)
    return min(recalls) if recalls else 0.0


def stratified_fold_indices(y, n_folds, rng):
    """
    Split indices into n_folds so each fold has roughly the same class distribution as y.
    Returns list of length n_folds: fold_chunks[f] = array of indices in validation fold f.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    fold_chunks = [[] for _ in range(n_folds)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_c = rng.permutation(idx_c)
        chunks_c = np.array_split(idx_c, n_folds)
        for f in range(n_folds):
            fold_chunks[f].append(chunks_c[f])
    for f in range(n_folds):
        fold_chunks[f] = np.concatenate(fold_chunks[f])
    return fold_chunks


def stratified_train_test_split(y, test_frac, rng):
    """
    Split indices so train and test keep roughly the same class ratio.
    Returns (train_idx, test_idx).
    """
    y = np.asarray(y)
    classes = np.unique(y)
    train_parts = []
    test_parts = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_c = rng.permutation(idx_c)
        n_c = len(idx_c)
        n_test_c = max(0, min(n_c, int(n_c * test_frac)))
        test_parts.append(idx_c[:n_test_c])
        train_parts.append(idx_c[n_test_c:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    return train_idx, test_idx


def main():
    # Set up the parser that reads command-line options (CSV path, target column, model settings, etc.).
    parser = argparse.ArgumentParser(
        description="RkCNN: Random k Conditional Nearest Neighbor classification from CSV."
    )
    # Input: CSV path and which column is the class. Or use --train and --validation for separate train/validation files.
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to CSV file (required unless --demo or --train/--validation)")
    parser.add_argument("--train", default=None, metavar="path", help="Train CSV path; use with --validation to fit on train and evaluate on validation (ignores csv_path and --test-frac)")
    parser.add_argument("--validation", default=None, metavar="path", help="Validation CSV path; use with --train for held-out evaluation")
    parser.add_argument("--target", default=None, help="Name of target (class) column")
    parser.add_argument("--target-index", type=int, default=None, help="0-based index of target column")
    # Model: k = neighbors per class, m = features per subset, r = how many top subsets, h = how many subsets to sample.
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbors per class (default: 3)")
    parser.add_argument("--m", type=int, default=None, help="Features per subset (default: ceil(sqrt(n_features)))")
    parser.add_argument("--r", type=int, default=200, help="Number of top subsets to use (default: 200). Paper: 200-300; use h >= 3*r for best results.")
    parser.add_argument("--h", type=int, default=600, help="Total random subsets to sample (default: 600). Paper: h = 3r to 10r for paper-like performance.")
    parser.add_argument("--test-frac", type=float, default=0.0, help="Fraction of data for test (0 = use all for train and evaluate on same)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scale", action="store_true", help="Standardize features (zero mean, unit variance) using training set only; often improves accuracy")
    parser.add_argument("--balance-weights", action="store_true", help="Apply class weights at prediction time to reduce majority-class bias (good for imbalanced data)")
    # Where to save each plot.
    parser.add_argument("--plot", default="clusters_2d.png", help="Output path for 2D cluster plot")
    parser.add_argument("--plot-3d", default="clusters_3d.png", help="Output path for 3D cluster plot")
    parser.add_argument("--plot-accuracy", default="accuracy_by_class.png", help="Output path for per-class accuracy bar chart")
    parser.add_argument("--plot-separation", default="separation_scores.png", help="Output path for separation score by feature bar chart")
    parser.add_argument("--plot-top", default="clusters_top5_features.png", help="Output path for pairwise scatter of top 5 features")
    parser.add_argument("--plot-max-features", type=int, default=15, help="Max features to show in separation bar chart (default: 15); use to avoid huge plots with many features")
    # Demo: run on built-in data (generated via rkcnn_data).
    parser.add_argument("--demo", action="store_true", help="Run on built-in example data (no CSV needed)")
    parser.add_argument("--demo-clusters", type=int, default=3, help="Number of clusters in demo mode (2, 3, 5, or any >= 2)")
    parser.add_argument("--tune", action="store_true", help="Run parameter tuning: try grid of (k,m,r,h), report best accuracy and params (no plots)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for CV when tuning with --train/--validation (default: 5)")
    parser.add_argument("--quiet", action="store_true", help="Less progress output")
    args = parser.parse_args()

    verbose = not args.quiet
    # If the user asked for tuning but did not set a test fraction, use 20% for testing.
    if args.tune and args.test_frac <= 0:
        args.test_frac = 0.2

    # Separate train and validation files: fit on train, evaluate on validation (ignores csv_path and --test-frac).
    use_separate_train_valid = (args.train is not None) and (args.validation is not None)
    if (args.train is not None) != (args.validation is not None):
        print("Error: both --train and --validation are required when using separate files.", file=sys.stderr)
        sys.exit(1)

    if use_separate_train_valid:
        print(f"Loading train CSV: {args.train}")
        X_train, y_train, feature_names, target_name = load_csv_for_classification(
            args.train, target_column=args.target, target_index=args.target_index)
        print(f"  Rows: {X_train.shape[0]}, Features: {X_train.shape[1]}, Target column: {target_name}")
        print("  Class counts:", dict((str(c), int(cnt)) for c, cnt in Counter(y_train).items()))
        print(f"Loading validation CSV: {args.validation}")
        X_test, y_test, _, _ = load_csv_for_classification(
            args.validation, target_column=args.target, target_index=args.target_index)
        print(f"  Rows: {X_test.shape[0]}, Features: {X_test.shape[1]}, Target column: {target_name}")
        print("  Class counts:", dict((str(c), int(cnt)) for c, cnt in Counter(y_test).items()))
        n_features = X_train.shape[1]
    else:
        # Get the data: either from demo (built-in clusters) or by loading the user's CSV.
        if args.demo:
            from rkcnn_data import generate_demo_clusters
            rng = np.random.default_rng(args.seed)
            n_clusters = max(2, args.demo_clusters)
            X, y = generate_demo_clusters(n_samples=100, n_features=10, n_clusters=n_clusters, rng=rng)
            feature_names = [f"f{i}" for i in range(X.shape[1])]
            target_name = "class"
            print("Running in demo mode: 100 samples, 10 features, {} classes (well-separated clusters).".format(n_clusters))
            print("  Target column: class. Features: f0..f9.")
        else:
            if not args.csv_path:
                print("Error: provide a CSV file path or use --demo.", file=sys.stderr)
                sys.exit(1)
            print(f"Loading CSV: {args.csv_path}")
            X, y, feature_names, target_name = load_csv_for_classification(
                args.csv_path,
                target_column=args.target,
                target_index=args.target_index,
            )
            print(f"  Rows: {X.shape[0]}, Features: {X.shape[1]}, Target column: {target_name}")
            print("  Class counts:", dict((str(c), int(cnt)) for c, cnt in Counter(y).items()))

        n_features = X.shape[1]

        # Split data into train and test if the user set a test fraction (stratified by class).
        if args.test_frac > 0 and args.test_frac < 1:
            rng = np.random.default_rng(args.seed)
            train_idx, test_idx = stratified_train_test_split(y, args.test_frac, rng)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            if verbose:
                print(f"  Train: {len(train_idx)}, Test: {len(test_idx)} (stratified split)")
        else:
            # No split: use all data for both training and evaluation.
            X_train, y_train = X, y
            X_test, y_test = X, y

    # Optional feature scaling: standardize using training set only (often improves validation accuracy).
    if args.scale:
        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0) + 1e-10
        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train
        if verbose:
            print("  Features standardized (train mean/std applied to train and validation).")

    # Set the number of features per subset (m); if not given, use the square root of the number of features.
    m = args.m if args.m is not None else max(1, int(np.ceil(np.sqrt(n_features))))
    if m > n_features:
        m = n_features
    if args.h < args.r:
        args.h = args.r
    if args.r > args.h:
        args.r = args.h

    # If the user asked for tuning: try each combination of (k, m, r, h), pick the best, print it, and exit.
    if args.tune:
        n_features = X_train.shape[1]
        m_default = max(1, int(np.ceil(np.sqrt(n_features))))
        k_vals = TUNE_K if isinstance(TUNE_K, (list, tuple)) else [TUNE_K]
        if TUNE_M is not None:
            m_vals = TUNE_M if isinstance(TUNE_M, (list, tuple)) else [TUNE_M]
        else:
            # Try several m around sqrt(n_features): smaller can help with noisy features, larger can capture more signal.
            m_vals = sorted(set([max(1, m_default // 2), m_default, min(2 * m_default, n_features)]))
        r_vals = TUNE_R if isinstance(TUNE_R, (list, tuple)) else [TUNE_R]
        if TUNE_H is None:
            # Try multiple pool sizes per r: 3*r, 5*r, 10*r (paper suggests h = 3r to 10r).
            combos = [(k, m, r, h) for k in k_vals for m in m_vals for r in r_vals for h in [3 * r, 5 * r, 10 * r] if h >= r]
        else:
            h_vals = TUNE_H if isinstance(TUNE_H, (list, tuple)) else [TUNE_H]
            combos = [(k, m, r, h) for k in k_vals for m in m_vals for r in r_vals for h in h_vals if h >= r]
        if not combos:
            print("No valid (k,m,r,h) combinations in tuning grid.", file=sys.stderr)
            sys.exit(1)
        if use_separate_train_valid:
            n_folds = max(2, min(args.cv_folds, X_train.shape[0]))
            print("\nTuning over {} combinations ({}-fold CV on training set only; validation held out for final evaluation)...".format(len(combos), n_folds))
        else:
            print("\nTuning over {} combinations (test-frac={})...".format(len(combos), args.test_frac))
        results = []
        if use_separate_train_valid:
            n_train = X_train.shape[0]
            rng_cv = np.random.default_rng(args.seed)
            fold_chunks = stratified_fold_indices(y_train, n_folds, rng_cv)
            for idx, (k, m, r, h) in enumerate(combos):
                m_use = min(m, n_features) if m > n_features else m
                fold_bal_accs = []
                fold_min_recalls = []
                for f in range(n_folds):
                    val_idx = fold_chunks[f]
                    train_idx = np.concatenate([fold_chunks[i] for i in range(n_folds) if i != f])
                    X_fit = X_train[train_idx]
                    y_fit = y_train[train_idx]
                    X_val = X_train[val_idx]
                    y_val = y_train[val_idx]
                    fit_f = rkcnn_fit(X_fit, y_fit, k=k, m=m_use, r=r, h=h, random_state=args.seed + f, verbose=False, use_class_weights=args.balance_weights)
                    y_pred_f, _ = rkcnn_predict(fit_f, X_val, verbose=False)
                    fold_bal_accs.append(balanced_accuracy(y_val, y_pred_f, fit_f["classes"]))
                    fold_min_recalls.append(min_per_class_recall(y_val, y_pred_f, fit_f["classes"]))
                cv_bal_acc = np.mean(fold_bal_accs)
                cv_min_recall = np.mean(fold_min_recalls)
                results.append((k, m_use, r, h, cv_bal_acc, cv_min_recall))
                print("  [{}/{}] k={} m={} r={} h={} -> CV balanced accuracy={:.4f} (min recall={:.2f})".format(idx + 1, len(combos), k, m_use, r, h, cv_bal_acc, cv_min_recall))
        else:
            for idx, (k, m, r, h) in enumerate(combos):
                m_use = min(m, n_features) if m > n_features else m
                fit_result = rkcnn_fit(X_train, y_train, k=k, m=m_use, r=r, h=h, random_state=args.seed, verbose=False, use_class_weights=args.balance_weights)
                y_pred_loop, _ = rkcnn_predict(fit_result, X_test, verbose=False)
                bal_acc = balanced_accuracy(y_test, y_pred_loop, fit_result["classes"])
                results.append((k, m_use, r, h, bal_acc))
                print("  [{}/{}] k={} m={} r={} h={} -> balanced accuracy={:.4f}".format(idx + 1, len(combos), k, m_use, r, h, bal_acc))
        if use_separate_train_valid:
            valid = [r for r in results if r[5] >= 0.10]
            best = max(valid, key=lambda r: r[4]) if valid else max(results, key=lambda r: r[4])
            if best[5] < 0.10:
                print("\n  (No combo had min per-class recall >= 10%; chosen best by CV balanced accuracy.)")
        else:
            best = max(results, key=lambda row: row[4])
        if use_separate_train_valid:
            print("\n--- Best parameters (by {}-fold CV balanced accuracy on training set) ---".format(n_folds))
            print("  k={}, m={}, r={}, h={} -> CV balanced accuracy={:.4f} (min recall={:.2f})".format(best[0], best[1], best[2], best[3], best[4], best[5]))
            fit_result = rkcnn_fit(X_train, y_train, k=best[0], m=best[1], r=best[2], h=best[3], random_state=args.seed, verbose=False, use_class_weights=args.balance_weights)
            y_pred_valid, _ = rkcnn_predict(fit_result, X_test, verbose=False)
            acc_valid = accuracy(y_test, y_pred_valid)
            bal_acc_valid = balanced_accuracy(y_test, y_pred_valid, fit_result["classes"])
            print("  Validation accuracy (overall): {:.4f}".format(acc_valid))
            print("  Validation balanced accuracy: {:.4f}".format(bal_acc_valid))
            args.k, args.m, args.r, args.h = best[0], best[1], best[2], best[3]
            m = best[1]
            target_for_cmd = " --target {}".format(args.target) if args.target else ""
            scale_cmd = " --scale" if args.scale else ""
            balance_cmd = " --balance-weights" if args.balance_weights else ""
            print("  Reproduce (no tune): python rkcnn.py --train {} --validation {}{}{}{} --k {} --m {} --r {} --h {}".format(
                args.train, args.validation, target_for_cmd, scale_cmd, balance_cmd, best[0], best[1], best[2], best[3]))
            print("  Reproduce (tune): python rkcnn.py --train {} --validation {}{}{}{} --tune --cv-folds {}".format(
                args.train, args.validation, target_for_cmd, scale_cmd, balance_cmd, args.cv_folds))
            print("  Running normal fit/predict with best params to write training/ and validation/ outputs...")
        else:
            print("\n--- Best parameters ---")
            print("  k={}, m={}, r={}, h={} -> balanced accuracy={:.4f}".format(best[0], best[1], best[2], best[3], best[4]))
            scale_cmd = " --scale" if args.scale else ""
            if args.demo:
                print("  Reproduce: python rkcnn.py --demo --demo-clusters {}{} --k {} --m {} --r {} --h {} --test-frac {}".format(
                    args.demo_clusters, scale_cmd, best[0], best[1], best[2], best[3], args.test_frac))
            else:
                target_for_cmd = " --target {}".format(args.target) if args.target else ""
                print("  Reproduce: python rkcnn.py {}{}{} --k {} --m {} --r {} --h {} --test-frac {}".format(
                    args.csv_path, target_for_cmd, scale_cmd, best[0], best[1], best[2], best[3], args.test_frac))
            sys.exit(0)

    # Normal run: fit the model on the training data, then predict on the test data.
    print("\nParameters: k={}, m={}, r={}, h={}".format(args.k, m, args.r, args.h))
    print("\nFitting RkCNN...")
    fit_result = rkcnn_fit(
        X_train, y_train,
        k=args.k, m=m, r=args.r, h=args.h,
        random_state=args.seed,
        verbose=verbose,
        use_class_weights=args.balance_weights,
    )
    print("  Fit done.\nPredicting...")
    y_pred, proba = rkcnn_predict(fit_result, X_test, verbose=verbose)
    print("  Predictions done.")

    # Compute accuracy and the confusion table (true class vs predicted class).
    acc = accuracy(y_test, y_pred)
    classes = fit_result["classes"]
    cm = confusion_matrix(y_test, y_pred, classes)
    n_features = X_test.shape[1]

    if use_separate_train_valid:
        # Separate train/validation mode: create training/ and validation/ folders and save results and plots in each.
        train_dir = "training"
        valid_dir = "validation"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

        # Top feature indices from TRAINING set only, so validation plots use the same 2D/3D space and don't collapse to one point.
        scores_train = separation_scores_per_feature(X_train, y_train)
        top2_t = np.argsort(-scores_train)[: min(2, n_features)]
        top3_t = np.argsort(-scores_train)[: min(3, n_features)]
        top5_t = np.argsort(-scores_train)[: min(5, n_features)]

        # --- Validation results and plots (use training's top indices for 2D/3D so validation points have spread) ---
        bal_acc = balanced_accuracy(y_test, y_pred, classes)
        print("\n--- Validation results ---")
        print("Accuracy (overall): {:.4f}".format(acc))
        print("Balanced accuracy: {:.4f}".format(bal_acc))
        print("Confusion matrix (rows = true class, columns = predicted class):")
        print("  Classes (order):", [str(c) for c in classes])
        print(cm)
        valid_results_path = os.path.join(valid_dir, "results.txt")
        with open(valid_results_path, "w") as f:
            f.write("Validation results\n")
            f.write("Accuracy (overall): {:.4f}\n".format(acc))
            f.write("Balanced accuracy: {:.4f}\n".format(bal_acc))
            f.write("Confusion matrix (rows = true class, columns = predicted class):\n")
            f.write("  Classes (order): {}\n".format([str(c) for c in classes]))
            f.write("{}\n".format(cm))
        print("  Results saved: {}".format(valid_results_path))
        scores_valid = separation_scores_per_feature(X_test, y_test)
        print("\nPlotting validation (separation score by feature)...")
        plot_separation_scores(scores_valid, feature_names=feature_names, filepath=os.path.join(valid_dir, "separation_scores.png"), max_features=args.plot_max_features)
        print("Plotting validation (2D clusters)...")
        plot_clusters_2d(X_test, y_true=y_test, y_pred=y_pred, filepath=os.path.join(valid_dir, "clusters_2d.png"), title="RkCNN clusters (2D) - validation", feature_names=feature_names, top_two_indices=top2_t)
        print("Plotting validation (3D clusters)...")
        plot_clusters_3d(X_test, y_true=y_test, y_pred=y_pred, filepath=os.path.join(valid_dir, "clusters_3d.png"), title="RkCNN clusters (3D) - validation", feature_names=feature_names, top_three_indices=top3_t)
        print("Plotting validation (pairwise scatter)...")
        plot_clusters_top_features_pairwise(X_test, y_true=y_test, y_pred=y_pred, top_indices=top5_t, feature_names=feature_names, filepath=os.path.join(valid_dir, "clusters_top5_features.png"), n_top=5)
        print("Plotting validation (per-class accuracy)...")
        plot_per_class_accuracy(y_test, y_pred, classes, filepath=os.path.join(valid_dir, "accuracy_by_class.png"), title="Accuracy by class (validation)")

        # --- Training results and plots (predict on training set) ---
        y_pred_train, _ = rkcnn_predict(fit_result, X_train, verbose=verbose)
        acc_train = accuracy(y_train, y_pred_train)
        bal_acc_train = balanced_accuracy(y_train, y_pred_train, classes)
        cm_train = confusion_matrix(y_train, y_pred_train, classes)
        print("\n--- Training results ---")
        print("Accuracy (overall): {:.4f}".format(acc_train))
        print("Balanced accuracy: {:.4f}".format(bal_acc_train))
        print("Confusion matrix (rows = true class, columns = predicted class):")
        print("  Classes (order):", [str(c) for c in classes])
        print(cm_train)
        train_results_path = os.path.join(train_dir, "results.txt")
        with open(train_results_path, "w") as f:
            f.write("Training results\n")
            f.write("Accuracy (overall): {:.4f}\n".format(acc_train))
            f.write("Balanced accuracy: {:.4f}\n".format(bal_acc_train))
            f.write("Confusion matrix (rows = true class, columns = predicted class):\n")
            f.write("  Classes (order): {}\n".format([str(c) for c in classes]))
            f.write("{}\n".format(cm_train))
        print("  Results saved: {}".format(train_results_path))
        print("\nPlotting training (separation score by feature)...")
        plot_separation_scores(scores_train, feature_names=feature_names, filepath=os.path.join(train_dir, "separation_scores.png"), max_features=args.plot_max_features)
        print("Plotting training (2D clusters)...")
        plot_clusters_2d(X_train, y_true=y_train, y_pred=y_pred_train, filepath=os.path.join(train_dir, "clusters_2d.png"), title="RkCNN clusters (2D) - training", feature_names=feature_names, top_two_indices=top2_t)
        print("Plotting training (3D clusters)...")
        plot_clusters_3d(X_train, y_true=y_train, y_pred=y_pred_train, filepath=os.path.join(train_dir, "clusters_3d.png"), title="RkCNN clusters (3D) - training", feature_names=feature_names, top_three_indices=top3_t)
        print("Plotting training (pairwise scatter)...")
        plot_clusters_top_features_pairwise(X_train, y_true=y_train, y_pred=y_pred_train, top_indices=top5_t, feature_names=feature_names, filepath=os.path.join(train_dir, "clusters_top5_features.png"), n_top=5)
        print("Plotting training (per-class accuracy)...")
        plot_per_class_accuracy(y_train, y_pred_train, classes, filepath=os.path.join(train_dir, "accuracy_by_class.png"), title="Accuracy by class (training)")
        print("Done. Outputs in {} and {}.".format(train_dir, valid_dir))
    else:
        # Single CSV or demo: one "Results" block and plots to current dir or user paths.
        bal_acc = balanced_accuracy(y_test, y_pred, classes)
        print("\n--- Results ---")
        print("Accuracy (overall): {:.4f}".format(acc))
        print("Balanced accuracy: {:.4f}".format(bal_acc))
        print("Confusion matrix (rows = true class, columns = predicted class):")
        print("  Classes (order):", [str(c) for c in classes])
        print(cm)
        scores = separation_scores_per_feature(X_test, y_test)
        top2 = np.argsort(-scores)[: min(2, n_features)]
        top3 = np.argsort(-scores)[: min(3, n_features)]
        top5 = np.argsort(-scores)[: min(5, n_features)]
        print("\nPlotting separation score by feature...")
        plot_separation_scores(scores, feature_names=feature_names, filepath=args.plot_separation, max_features=args.plot_max_features)
        print("Plotting 2D clusters (top 2 features by separation)...")
        plot_clusters_2d(X_test, y_true=y_test, y_pred=y_pred, filepath=args.plot, title="RkCNN clusters (2D)", feature_names=feature_names, top_two_indices=top2)
        print("Plotting 3D clusters (top 3 features by separation)...")
        plot_clusters_3d(X_test, y_true=y_test, y_pred=y_pred, filepath=args.plot_3d, title="RkCNN clusters (3D)", feature_names=feature_names, top_three_indices=top3)
        print("Plotting pairwise scatter (top 5 features by separation)...")
        plot_clusters_top_features_pairwise(X_test, y_true=y_test, y_pred=y_pred, top_indices=top5, feature_names=feature_names, filepath=args.plot_top, n_top=5)
        print("Plotting per-class accuracy (bar chart)...")
        plot_per_class_accuracy(y_test, y_pred, classes, filepath=args.plot_accuracy, title="Accuracy by class (percentage correct)")
        print("Done.")


# When this file is run as a script (e.g. python rkcnn.py data.csv), call main().
if __name__ == "__main__":
    main()
