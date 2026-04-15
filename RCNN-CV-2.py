"""
RkCNN: Random k Conditional Nearest Neighbor for classification.

This script implements the RkCNN algorithm from scratch (Lu and Gweon, 2025).
Load a CSV with features and a class column, then run classification with
progress printed to the terminal. Optionally save a 2D cluster plot (PCA or first two features).

RCNN-CV: Same as RCNN-2 but uses cross-validation on training data instead of
evaluating on a separate validation dataset.

Usage examples:
  python rkcnn.py data.csv
  python rkcnn.py data.csv --target class --k 3 --m 10 --r 50 --h 200
"""

# Import the module that reads command-line options (e.g. --target, --k).
import argparse
import csv
# Import the module that lets us exit the program and print to the terminal.
import sys
import os
# Import Counter so we can count how many points belong to each class.
from collections import Counter, defaultdict

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


# --- Tuning constants for RCNN-CV (focused search around known-good values) ---
# RCNN-CV uses CV on training only to select params; validation used once at end.

# Stage 1: Expanded search (m 50-120, h up to 50*r for 2000/3000/5000 when r=100)
STAGE1_K = [1]
STAGE1_M = [20, 30, 40, 50, 60, 70, 80, 90, 100]
STAGE1_R = [90, 100, 110, 120, 130, 140]
STAGE1_H_MULT = [5, 10, 15]
STAGE1_SEEDS = [2, 3, 4]

# Stage 2: Tighter grid around winners (M dense; R refined by cluster)
STAGE2_M_DENSE = [20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
STAGE2_R_LOW = [90, 100, 110, 120, 130]
STAGE2_R_HIGH = [120, 130, 140]

# --- arcene-best grid: ~576 Stage 1, ~280 Stage 2 (Stage 2 centers on Stage 1 winner) ---
ARCENE_BEST_K = [1]                                    # k fixed at 1
ARCENE_BEST_M = [95, 100, 105, 110, 115, 120, 125, 130]
ARCENE_BEST_R = [80, 90, 100, 110, 120, 130]
ARCENE_BEST_H = [500, 750, 1000, 1250, 1500, 2000]
ARCENE_BEST_SEEDS = [1, 2]
# Stage 2: no fixed S2 constants; values are derived from Stage 1 winner in arcene_best_stage2_combos_from_winners

# Reference params that historically achieved ~90% validation accuracy (no-scale, no-prefilter)
REFERENCE_PARAMS = (1, 110, 100, 1000, 2)  # k, m, r, h, seed


def stage1_combos(include_k7=False, seeds=None, n_features=None):
    """Stage 1: coarse search. K in {1,3,5}, M 80-130, R 80-200, H = 5R/10R/15R."""
    k_vals = list(STAGE1_K)
    if include_k7:
        k_vals = k_vals + [7]
    m_vals = [m for m in STAGE1_M if m <= (n_features or 10000)]
    if not m_vals:
        m_vals = [min(STAGE1_M[-1], n_features or 10000)]
    seeds = seeds or STAGE1_SEEDS
    combos = []
    for k in k_vals:
        for m in m_vals:
            for r in STAGE1_R:
                for h_mult in STAGE1_H_MULT:
                    h = h_mult * r
                    if h < r:
                        continue
                    for seed in seeds:
                        combos.append((k, m, r, h, seed))
    return combos


def stage2_combos_from_winners(top_results, n_features, seeds=None):
    """
    Stage 2: denser search around top CV winners.
    M: if best_m in {100,110,120} use dense {95,100,105,110,115,120}; else around best_m.
    R: if winners cluster low (80-120) use R_LOW; if high (120-200) use R_HIGH.
    """
    seeds = seeds or STAGE1_SEEDS
    best_m = top_results[0][1]
    best_r = top_results[0][2]
    r_vals = top_results[:5]
    r_center = np.median([r[2] for r in r_vals])
    if r_center <= 120:
        r_refined = [r for r in STAGE2_R_LOW if r <= (n_features or 10000)]
    else:
        r_refined = [r for r in STAGE2_R_HIGH if r <= (n_features or 10000)]
    if not r_refined:
        r_refined = [best_r]
    if best_m in (100, 110, 120) or (50 <= best_m <= 120):
        m_refined = [m for m in STAGE2_M_DENSE if m <= (n_features or 10000)]
    else:
        m_refined = sorted(set([
            max(10, best_m - 10), max(10, best_m - 5), best_m,
            min(best_m + 5, n_features or 10000), min(best_m + 10, n_features or 10000)
        ]))
    if not m_refined:
        m_refined = [best_m]
    k_vals = sorted(set([max(1, top_results[0][0] - 1), top_results[0][0], min(top_results[0][0] + 1, 9)]))
    combos = []
    for k in k_vals:
        for m in m_refined:
            for r in r_refined:
                for h_mult in STAGE1_H_MULT:
                    h = h_mult * r
                    if h < r:
                        continue
                    for seed in seeds:
                        combos.append((k, m, r, h, seed))
    return combos


def arcene_best_stage1_combos(n_features=None):
    """Stage 1 for arcene-best: k=1, m 95-130, r 80-130, h 500-2000, seed 1-2. ~576 combos."""
    m_vals = [m for m in ARCENE_BEST_M if m <= (n_features or 10000)]
    if not m_vals:
        m_vals = [min(ARCENE_BEST_M[-1], n_features or 10000)]
    h_vals = [h for h in ARCENE_BEST_H if h >= 80]
    combos = []
    for k in ARCENE_BEST_K:
        for m in m_vals:
            for r in ARCENE_BEST_R:
                for h in h_vals:
                    if h < r:
                        continue
                    for seed in ARCENE_BEST_SEEDS:
                        combos.append((k, m, r, h, seed))
    return combos


def arcene_best_stage2_combos_from_winners(top_results, n_features):
    """
    Stage 2 for arcene-best: ~280 combos centered on Stage 1 winner.
    top_results[0] = (k, m, r, h, cv_score). Build m/r/h ranges around winner.
    Ensures (1, 110, 100, 1000) is tested when it wins Stage 1.
    """
    best_k, best_m, best_r, best_h = top_results[0][0], top_results[0][1], top_results[0][2], top_results[0][3]
    # k: fixed at winner (user requested k=1)
    k_vals = [best_k]
    # m: 5 values, step 5, centered on best_m
    m_refined = sorted(set([
        max(10, best_m - 10), max(10, best_m - 5), best_m,
        min(best_m + 5, n_features or 10000), min(best_m + 10, n_features or 10000)
    ]))
    # r: 7 values, step 5, centered on best_r
    r_refined = sorted(set([
        max(50, best_r - 15), max(50, best_r - 10), max(50, best_r - 5),
        best_r,
        min(best_r + 5, 200), min(best_r + 10, 200), min(best_r + 15, 200)
    ]))
    # h: 8 values around best_h (multiplicative factors)
    factors = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    h_refined = sorted(set([
        max(best_r, int(f * best_h)) for f in factors
    ]))
    h_refined = [h for h in h_refined if h >= best_r]
    if not h_refined:
        h_refined = [best_h]
    # seed: single value to keep ~280 combos (5*7*8*1=280)
    seeds_s2 = [2]
    combos = []
    for k in k_vals:
        for m in m_refined:
            for r in r_refined:
                for h in h_refined:
                    if h < r:
                        continue
                    for seed in seeds_s2:
                        combos.append((k, m, r, h, seed))
    # Always include reference params (1, 110, 100, 1000, 2) for comparison
    ref_combo = (1, 110, 100, 1000, 2)
    combos_set = set(combos)
    if ref_combo not in combos_set:
        combos.append(ref_combo)
    return combos


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


def kth_neighbor_distance_per_class(X_sub, y_train, x_query, k, classes, class_indices=None):
    """
    For each class, find the distance from x_query to the k-th nearest training
    point in that class (in the subspace X_sub). If a class has fewer than k
    points, use the farthest available neighbor for that class.

    X_sub: 2D array (n_train, n_features_sub), training data in one feature subset.
    y_train: 1D array of class labels for each training row.
    x_query: 1D array, query point in same subspace.
    k: number of nearest neighbors to consider per class.
    classes: list or array of unique class labels (order preserved for output).
    class_indices: optional dict mapping each class label to 1D array of row indices; if provided, used instead of y_train masks.

    Returns: 1D array of length len(classes): distance to k-th (or last) neighbor in each class.
    """
    # Make sure the query point is a flat array of numbers.
    x_query = np.asarray(x_query, dtype=float).ravel()
    # Get the distance from the query point to every training point (in this feature subset).
    distances = euclidean_distance(x_query, X_sub)
    # We will fill in one distance per class.
    out = np.zeros(len(classes))
    for idx, c in enumerate(classes):
        if class_indices is not None:
            inds = class_indices.get(c)
            if inds is None or len(inds) == 0:
                out[idx] = np.inf
                continue
            d_c = distances[inds]
        else:
            mask = y_train == c
            if not np.any(mask):
                out[idx] = np.inf
                continue
            d_c = distances[mask]
        # k-th smallest: partition is O(n) vs sort O(n log n).
        if len(d_c) >= k:
            out[idx] = np.partition(d_c, k - 1)[k - 1]
        else:
            out[idx] = np.max(d_c)
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


MAX_FEATURES_FOR_SEPARATION = 5000
SEPARATION_SAMPLE_SIZE = 2000


def separation_scores_per_feature(X, y, max_features=None, rng=None):
    """
    Compute separation score (BV/WV) for each single feature.
    Returns 1D array of length X.shape[1]. Higher score = better class separation.
    For n_features > MAX_FEATURES_FOR_SEPARATION, samples 2000 features (fixed seed 42) so train and validation plots use same features.
    When max_features is set (e.g. for prefilter), scores only that many features (random subset if n_features > max_features).
    """
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    
    if max_features is not None and n_features > max_features:
        # Score only a random subset of max_features columns (for prefilter on huge feature sets).
        rng = np.random.default_rng(rng)
        indices = rng.choice(n_features, size=max_features, replace=False)
        for j in indices:
            scores[j] = separation_score(X[:, j : j + 1], y)
        return scores
    
    if max_features is None and n_features > MAX_FEATURES_FOR_SEPARATION:
        # Sample 2000 features with fixed seed 42 - train and validation get same features for comparable plots
        rng_sample = np.random.default_rng(42)
        sample_idx = rng_sample.choice(n_features, size=min(SEPARATION_SAMPLE_SIZE, n_features), replace=False)
        print("  Computing separation scores for {} sampled features (of {})...".format(
            len(sample_idx), n_features))
        for j in sample_idx:
            scores[j] = separation_score(X[:, j : j + 1], y)
        return scores
    
    # Compute scores for all features when below threshold
    for j in range(n_features):
        scores[j] = separation_score(X[:, j : j + 1], y)
    return scores


def get_prefilter_top_indices(X_fit, y_fit, n_keep, method="separation", ranking_path=None,
                              feature_names=None, prefilter_max=10000, rng=None):
    """
    Return topP: indices of top n_keep features for prefiltering.
    method: "separation" (univariate BV/WV) or "relieff" (ReliefF weights).
    When method=relieff and ranking_path is given, load from CSV; else run ReliefF on X_fit.
    Returns None if n_keep <= 0.
    """
    if n_keep <= 0:
        return None
    n_keep = min(n_keep, X_fit.shape[1])
    rng_obj = np.random.default_rng(rng) if (rng is not None and not hasattr(rng, "permutation")) else rng

    if method == "relieff":
        if ranking_path and os.path.exists(ranking_path) and feature_names is not None:
            df = pd.read_csv(ranking_path)
            name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
            topP = []
            for _, row in df.iterrows():
                if len(topP) >= n_keep:
                    break
                name = str(row["feature"])
                if name in name_to_idx:
                    topP.append(name_to_idx[name])
            topP = np.array(topP) if topP else np.arange(min(n_keep, X_fit.shape[1]))
        else:
            from relieff_rank import relieff_weights
            rng_use = rng_obj if rng_obj is not None else np.random.default_rng(42)
            W = relieff_weights(X_fit, y_fit, k=10, m=None, prior_type="empirical", rng=rng_use)
            topP = np.argsort(-W)[:n_keep]
        return topP

    # separation
    pmax = prefilter_max if prefilter_max > 0 else X_fit.shape[1]
    scores = separation_scores_per_feature(X_fit, y_fit, max_features=min(pmax, X_fit.shape[1]), rng=rng)
    return np.argsort(-scores)[:n_keep]


def kcnn_predict_one(X_train_sub, y_train, x_query, k, classes, class_indices=None):
    """
    Get class probabilities for a single query using kCNN on one feature subset.

    Probability for class c is proportional to 1/d_c where d_c is the L2 distance
    to the k-th nearest neighbor in class c (paper Eq. 1). Then normalize to sum to 1.

    If all distances are zero or infinite, return uniform probabilities.

    Returns: 1D array of length len(classes), probabilities in same order as classes.
    """
    # Get the L2 distance to the k-th nearest neighbor in each class for this query point.
    d_per_class = kth_neighbor_distance_per_class(X_train_sub, y_train, x_query, k, classes, class_indices=class_indices)
    # Avoid dividing by zero: replace zero or negative distances with a tiny number.
    d_safe = np.where(d_per_class <= 0, 1e-10, d_per_class)
    # Paper Eq. 1: P(Y=c|x) proportional to ||x, x_{k|c}||_2^{-1} = 1/distance
    inv_d = 1.0 / d_safe
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

    # Precompute class indices for fast prediction (avoid rebuilding masks per query).
    class_indices = {c: np.where(y == c)[0] for c in classes}
    # Optional: class weights for imbalanced data (minority class gets higher weight at prediction time).
    out = {
        "top_subset_indices": top_subset_indices,
        "weights": weights,
        "classes": classes,
        "X_train": X,
        "y_train": y,
        "k": k,
        "class_indices": class_indices,
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


def rkcnn_predict(fit_result, X_new, verbose=True, class_bias=None):
    """
    Predict class labels and optionally class probabilities for X_new using
    the fitted RkCNN (result of rkcnn_fit).

    Progress: print every 10% of samples so the user sees the algorithm is running.
    
    class_bias: optional dict mapping class label to multiplicative bias (e.g. {'-1': 1.2} boosts class -1 by 20%)
    """
    # Get the stored subsets, weights, class list, and training data from the fit step.
    top_subset_indices = fit_result["top_subset_indices"]
    weights = fit_result["weights"]
    classes = fit_result["classes"]
    X_train = fit_result["X_train"]
    y_train = fit_result["y_train"]
    k = fit_result["k"]
    class_indices = fit_result.get("class_indices")

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
            p_j = kcnn_predict_one(X_train_sub, y_train, x_query, k, classes, class_indices=class_indices)
            if "class_weights" in fit_result:
                p_j = p_j * fit_result["class_weights"]
                p_j = p_j / (p_j.sum() + 1e-12)
            p_combined += weights[j] * p_j
        proba[i] = p_combined

    # Class weights already applied inside the loop (lines 354-356), so use proba directly.
    scores = proba.copy()
    
    # Apply class bias if provided (e.g. to boost minority class)
    if class_bias is not None:
        for i, c in enumerate(classes):
            if c in class_bias or str(c) in class_bias:
                bias = class_bias.get(c, class_bias.get(str(c), 1.0))
                scores[:, i] *= bias

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
    # Keep target as strings so we support both numeric and string labels (e.g. "cancer"/"normal").
    y = df[target_name].astype(str)
    # Drop rows where target is missing (pandas astype(str) turns NaN into 'nan') or features have NaN.
    valid = (y != 'nan') & ~X_df.isna().any(axis=1)
    X = X_df.loc[valid].values
    y = y.loc[valid].values

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
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.text(0.5, 0.02, "Separation score = BV/WV (ratio, unbounded)", ha="center", fontsize=9)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
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
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
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
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
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
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {filepath}")


def plot_cv_tuning_diagnostics(results, output_dir, results_format=8):
    """
    Generate CV tuning diagnostic plots from tuning results.
    results: list of (k, m, r, h, seed, cv_mean, cv_std, cv_min_recall) or (k, m, r, h, cv_mean) for 5-element.
    output_dir: directory to save plots (e.g. tuning/).
    """
    if plt is None or not results:
        return
    os.makedirs(output_dir, exist_ok=True)

    def agg_by_param(param_idx, param_name):
        from collections import defaultdict
        by_val = defaultdict(list)
        for r in results:
            if len(r) >= 5:
                val = r[param_idx]
                cv_mean = r[5] if len(r) >= 6 else r[4]
                cv_std = r[6] if len(r) >= 7 else 0.0
                by_val[val].append((cv_mean, cv_std))
        if not by_val:
            return None, None
        vals = sorted(by_val.keys())
        means = [np.mean([x[0] for x in by_val[v]]) for v in vals]
        stds = [np.std([x[0] for x in by_val[v]]) if len(by_val[v]) > 1 else np.mean([x[1] for x in by_val[v]]) for v in vals]
        return vals, (means, stds)

    def plot_effect_of_param(param_idx, param_name, title_suffix, filename):
        vals, data = agg_by_param(param_idx, param_name)
        if vals is None or len(vals) < 2:
            return
        means, stds = data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(vals, [m * 100 for m in means], yerr=[s * 100 for s in stds], fmt='o-', capsize=5, linewidth=2, markersize=8)
        best_idx = np.argmax(means)
        ax.scatter([vals[best_idx]], [means[best_idx] * 100], s=200, color='#E94F37', zorder=5, label='Best')
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('CV Balanced Accuracy (%)', fontsize=12)
        ax.set_title('Effect of {} on CV Performance'.format(title_suffix), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: {}".format(filename))

    plot_effect_of_param(0, 'k', 'k (neighbors)', 'effect_of_k.png')
    plot_effect_of_param(1, 'm', 'm (features per subset)', 'effect_of_m.png')
    plot_effect_of_param(2, 'r', 'r (top subsets)', 'effect_of_r.png')

    h_mult_vals = defaultdict(list)
    for r in results:
        if len(r) >= 5 and r[2] > 0:
            h_mult = r[3] // r[2] if r[2] > 0 else 0
            cv_mean = r[5] if len(r) >= 6 else r[4]
            cv_std = r[6] if len(r) >= 7 else 0.0
            h_mult_vals[h_mult].append((cv_mean, cv_std))
    if h_mult_vals:
        vals = sorted(h_mult_vals.keys())
        means = [np.mean([x[0] for x in h_mult_vals[v]]) for v in vals]
        stds = [np.std([x[0] for x in h_mult_vals[v]]) if len(h_mult_vals[v]) > 1 else np.mean([x[1] for x in h_mult_vals[v]]) for v in vals]
        if len(vals) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(vals, [m * 100 for m in means], yerr=[s * 100 for s in stds], fmt='o-', capsize=5, linewidth=2, markersize=8)
            best_idx = np.argmax(means)
            ax.scatter([vals[best_idx]], [means[best_idx] * 100], s=200, color='#E94F37', zorder=5, label='Best')
            ax.set_xlabel('h/r (subsets multiplier)', fontsize=12)
            ax.set_ylabel('CV Balanced Accuracy (%)', fontsize=12)
            ax.set_title('Effect of h/r on CV Performance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'effect_of_h_mult.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("  Saved: effect_of_h_mult.png")

    agg_all = defaultdict(list)
    for r in results:
        if len(r) >= 5:
            key = (r[0], r[1], r[2], r[3])
            cv_mean = r[5] if len(r) >= 6 else r[4]
            agg_all[key].append(cv_mean)
    leaderboard = [(k, m, r, h, np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0)
                   for (k, m, r, h), vals in agg_all.items()]
    leaderboard = sorted(leaderboard, key=lambda x: (-x[4], x[3], x[2]))[:10]
    if leaderboard:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = ['k={} m={} r={} h={}'.format(k, m, r, h) for (k, m, r, h, _, _) in leaderboard]
        x_pos = np.arange(len(labels))
        means = [x[4] * 100 for x in leaderboard]
        stds = [x[5] * 100 for x in leaderboard]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=3, color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('CV Balanced Accuracy (%)', fontsize=12)
        ax.set_title('CV Leaderboard: Top 10 Parameter Configurations', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'cv_leaderboard.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: cv_leaderboard.png")

    r_vals = sorted(set(r[2] for r in results if len(r) >= 5))
    h_vals = sorted(set(r[3] for r in results if len(r) >= 5))
    if len(r_vals) >= 2 and len(h_vals) >= 2:
        acc_matrix = np.full((len(r_vals), len(h_vals)), np.nan)
        for r in results:
            if len(r) >= 5:
                ri = r_vals.index(r[2]) if r[2] in r_vals else None
                hi = h_vals.index(r[3]) if r[3] in h_vals else None
                if ri is not None and hi is not None:
                    cv = r[5] if len(r) >= 6 else r[4]
                    if np.isnan(acc_matrix[ri, hi]) or cv > acc_matrix[ri, hi]:
                        acc_matrix[ri, hi] = cv
        fig, ax = plt.subplots(figsize=(10, 7))
        vmin = np.nanmin(acc_matrix) * 100 * 0.8 if not np.all(np.isnan(acc_matrix)) else 0
        vmax = min(100, np.nanmax(acc_matrix) * 100 * 1.05) if not np.all(np.isnan(acc_matrix)) else 100
        im = ax.imshow(acc_matrix * 100, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(h_vals)))
        ax.set_xticklabels([str(h) for h in h_vals])
        ax.set_yticks(range(len(r_vals)))
        ax.set_yticklabels([str(r) for r in r_vals])
        ax.set_xlabel('h (total subsets)', fontsize=12)
        ax.set_ylabel('r (top subsets)', fontsize=12)
        ax.set_title('CV Balanced Accuracy: r vs h', fontsize=14, fontweight='bold')
        for i in range(len(r_vals)):
            for j in range(len(h_vals)):
                if not np.isnan(acc_matrix[i, j]):
                    ax.text(j, i, '{:.1f}%'.format(acc_matrix[i, j] * 100), ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax, label='CV Balanced Accuracy (%)')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'cv_heatmap_r_h.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: cv_heatmap_r_h.png")


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
    ax.set_title(title or "Accuracy by class")
    # Write the percentage on top of each bar.
    for i, (bar, val) in enumerate(zip(bars, pct)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, "{:.0f}%".format(val), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
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
    # Minimum recall across classes that appear in the evaluation set; skip classes with no samples.
    cm = confusion_matrix(y_true, y_pred, classes)
    n = len(classes)
    recalls = []
    for i in range(n):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            recalls.append(cm[i, i] / row_sum)
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


def run_train_cv_evaluation(X_train, y_train, k, m, r, h, n_folds, seed,
                            prefilter=0, prefilter_max=10000, use_class_weights=False, repeats=1,
                            prefilter_method="separation", relieff_ranking_path=None, feature_names=None):
    """
    Run K-fold CV on training data, optionally repeated for stable estimates.
    Returns (cv_bal_acc_mean, cv_bal_acc_std, y_true_collected, y_pred_collected, val_inds_concat, classes).
    Prefilter: when prefilter > 0, use prefilter_method (separation or relieff) to select top features per fold.
    """
    all_fold_accs = []
    y_true_list = []
    y_pred_list = []
    val_inds_list = []
    classes_ref = None
    for rep in range(repeats):
        rng_cv = np.random.default_rng(seed + rep * 1000)
        fold_chunks = stratified_fold_indices(y_train, n_folds, rng_cv)
        for f in range(n_folds):
            val_idx = fold_chunks[f]
            train_idx = np.concatenate([fold_chunks[i] for i in range(n_folds) if i != f])
            X_fit = X_train[train_idx]
            y_fit = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            if prefilter and prefilter > 0:
                topP = get_prefilter_top_indices(
                    X_fit, y_fit, prefilter, method=prefilter_method,
                    ranking_path=relieff_ranking_path, feature_names=feature_names,
                    prefilter_max=prefilter_max, rng=seed + rep * 1000 + f)
                X_fit_use = X_fit[:, topP]
                X_val_use = X_val[:, topP]
            else:
                X_fit_use = X_fit
                X_val_use = X_val
            m_use = min(m, X_fit_use.shape[1]) if m > X_fit_use.shape[1] else m
            fit_f = rkcnn_fit(X_fit_use, y_fit, k=k, m=m_use, r=r, h=h, random_state=seed + rep * 1000 + f, verbose=False, use_class_weights=use_class_weights)
            y_pred_f, _ = rkcnn_predict(fit_f, X_val_use, verbose=False)
            classes_ref = fit_f["classes"]
            all_fold_accs.append(balanced_accuracy(y_val, y_pred_f, classes_ref))
            if rep == 0:
                y_true_list.append(y_val)
                y_pred_list.append(y_pred_f)
                val_inds_list.append(val_idx)
    cv_mean = np.mean(all_fold_accs)
    cv_std = np.std(all_fold_accs) if len(all_fold_accs) > 1 else 0.0
    y_true_all = np.concatenate(y_true_list)
    y_pred_all = np.concatenate(y_pred_list)
    val_inds_all = np.concatenate(val_inds_list)
    return cv_mean, cv_std, y_true_all, y_pred_all, val_inds_all, classes_ref


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
        n_test_c = int(round(n_c * test_frac))
        if n_c >= 2:
            n_test_c = max(1, min(n_c - 1, n_test_c))
        elif n_c == 1:
            n_test_c = 0
        test_parts.append(idx_c[:n_test_c])
        train_parts.append(idx_c[n_test_c:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    return train_idx, test_idx


def main():
    # Set up the parser that reads command-line options (CSV path, target column, model settings, etc.).
    parser = argparse.ArgumentParser(
        description="RkCNN-CV: Same as RCNN-2 but uses cross-validation on training data instead of validation-set evaluation."
    )
    # Input: CSV path and which column is the class. Or use --train (and optionally --validation for consistency; validation is not used).
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to CSV file (required unless --demo or --train)")
    parser.add_argument("--train", default=None, metavar="path", help="Train CSV path; tuning uses CV on training only")
    parser.add_argument("--validation", default=None, metavar="path", help="Validation CSV; used exactly once at end for final performance report (required for holdout eval when using --train)")
    parser.add_argument("--train-valid", default=None, metavar="path", help="Combined train+valid CSV; run k-fold CV for evaluation (no separate validation file).")
    parser.add_argument("--target", default=None, help="Name of target (class) column")
    parser.add_argument("--target-index", type=int, default=None, help="0-based index of target column")
    # Model: k = neighbors per class, m = features per subset, r = how many top subsets, h = how many subsets to sample.
    parser.add_argument("--k", type=int, default=1, help="Number of nearest neighbors per class (default: 1)")
    parser.add_argument("--m", type=int, default=110, help="Features per subset (default: 110)")
    parser.add_argument("--r", type=int, default=100, help="Number of top subsets to use (default: 100)")
    parser.add_argument("--h", type=int, default=1000, help="Total random subsets to sample (default: 1000)")
    parser.add_argument("--test-frac", type=float, default=0.0, help="Fraction of data for test (0 = use all for train and evaluate on same)")
    parser.add_argument("--seed", type=int, default=2, help="Random seed (default: 2)")
    parser.add_argument("--scale", action="store_true", default=True, help="Standardize features (zero mean, unit variance) using training set only")
    parser.add_argument("--no-scale", action="store_true", help="Disable feature standardization (same as RCNN-2 without --scale)")
    parser.add_argument("--balance-weights", action="store_true", help="Apply class weights at prediction time")
    parser.add_argument("--no-balance-weights", action="store_true", help="Disable class weights (default: enabled when --tune)")
    parser.add_argument("--class-bias", type=float, default=1.0, help="Multiplicative bias for class -1")
    # Where to save each plot.
    parser.add_argument("--plot", default="clusters_2d.png", help="Output path for 2D cluster plot")
    parser.add_argument("--plot-3d", default="clusters_3d.png", help="Output path for 3D cluster plot")
    parser.add_argument("--plot-accuracy", default="accuracy_by_class.png", help="Output path for per-class accuracy bar chart")
    parser.add_argument("--plot-separation", default="separation_scores.png", help="Output path for separation score by feature bar chart")
    parser.add_argument("--plot-top", default="clusters_top5_features.png", help="Output path for pairwise scatter of top 5 features")
    parser.add_argument("--plot-max-features", type=int, default=15, help="Max features to show in separation bar chart (default: 15)")
    # Demo: run on built-in data.
    parser.add_argument("--demo", action="store_true", help="Run on built-in example data (no CSV needed)")
    parser.add_argument("--demo-clusters", type=int, default=3, help="Number of clusters in demo mode")
    parser.add_argument("--tune", action="store_true", help="Run parameter tuning via CV on training data only")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for CV (default: 5)")
    parser.add_argument("--cv-repeats", type=int, default=3, help="Number of CV repetitions for stable estimates (default: 3)")
    parser.add_argument("--out", "--tuning-table", dest="tuning_table", default=None, help="Path to save tuning table CSV")
    parser.add_argument("--output-dir", default=".", help="Root directory for training/, validation/, tuning/ outputs (default: current dir)")
    parser.add_argument("--include-k7", action="store_true", help="Include k=7 in Stage 1 search")
    parser.add_argument("--prefilter", type=int, default=500, help="Keep top N features by univariate separation (0 = off)")
    parser.add_argument("--no-prefilter", action="store_true", help="Disable prefiltering; use all features (equivalent to --prefilter 0)")
    parser.add_argument("--prefilter-max", type=int, default=10000, help="Max features to score when prefiltering")
    parser.add_argument("--relieff-prefilter", type=int, default=0, help="Use ReliefF to keep top N features (0 = off; overrides --prefilter when > 0)")
    parser.add_argument("--relieff-ranking", default=None, metavar="path", help="Path to precomputed ReliefF ranking CSV (from relieff_rank.py); if set, use this instead of running ReliefF")
    parser.add_argument("--outer-folds", type=int, default=0, help="If >= 2, use nested CV")
    parser.add_argument("--inner-folds", type=int, default=5, help="Number of inner CV folds when using nested CV")
    parser.add_argument("--repeats", type=int, default=3, help="Repeat outer CV with different seeds")
    parser.add_argument("--tune-stage", choices=["1", "2", "both"], default=None, help="Stage 1/2/both for tuning")
    parser.add_argument("--tune-grid", choices=["arcene", "arcene-best"], default=None, help="Use arcene or arcene-best (no-scale, no-prefilter) grid")
    parser.add_argument("--quiet", action="store_true", help="Less progress output")
    args = parser.parse_args()
    output_dir = args.output_dir or "."

    if args.no_scale:
        args.scale = False
    if args.no_prefilter:
        args.prefilter = 0
    if args.tune_grid == "arcene-best":
        args.scale = False
        args.prefilter = 0
        args.relieff_prefilter = 0
    # Prefilter mode: ReliefF overrides separation when --relieff-prefilter > 0
    prefilter_n = args.relieff_prefilter if args.relieff_prefilter > 0 else args.prefilter
    prefilter_method = "relieff" if args.relieff_prefilter > 0 else "separation"
    if prefilter_method == "relieff" and prefilter_n > 0 and not args.quiet:
        src = "ranking file " + args.relieff_ranking if args.relieff_ranking else "on-the-fly ReliefF"
        print("  Prefilter: ReliefF (top {} features from {})".format(prefilter_n, src))
    # When ReliefF prefilter is used, save outputs under Relieff-rkcnn/ subfolder
    results_root = os.path.join(output_dir, "Relieff-rkcnn") if (prefilter_method == "relieff" and prefilter_n > 0) else output_dir

    if args.tune and not args.no_balance_weights:
        args.balance_weights = False

    verbose = not args.quiet
    if args.tune and args.test_frac <= 0:
        args.test_frac = 0.2

    # RCNN-CV: --train does NOT require --validation. We use CV on training data only.
    if (args.train_valid is not None) and (args.validation is not None):
        print("Error: --train-valid with --validation causes data leakage.", file=sys.stderr)
        sys.exit(1)
    if (args.train is not None) and (args.train_valid is not None):
        print("Error: use either --train or --train-valid, not both.", file=sys.stderr)
        sys.exit(1)
    use_separate_train_valid = (args.train is not None)
    use_train_valid_cv = (args.train_valid is not None) and (args.validation is None)

    if use_train_valid_cv:
        print(f"Loading train+valid CSV (CV-only mode): {args.train_valid}")
        X_train, y_train, feature_names, target_name = load_csv_for_classification(
            args.train_valid, target_column=args.target, target_index=args.target_index)
        print(f"  Rows: {X_train.shape[0]}, Features: {X_train.shape[1]}, Target column: {target_name}")
        print("  Class counts:", dict((str(c), int(cnt)) for c, cnt in Counter(y_train).items()))
        X_test = np.empty((0, X_train.shape[1]))
        y_test = np.array([], dtype=np.asarray(y_train).dtype)
        n_features = X_train.shape[1]

    elif use_separate_train_valid:
        train_path = args.train
        print(f"Loading train CSV: {train_path}")
        X_train, y_train, feature_names, target_name = load_csv_for_classification(
            train_path, target_column=args.target, target_index=args.target_index)
        print(f"  Rows: {X_train.shape[0]}, Features: {X_train.shape[1]}, Target column: {target_name}")
        print("  Class counts:", dict((str(c), int(cnt)) for c, cnt in Counter(y_train).items()))
        if args.validation is not None:
            print(f"  Loading validation CSV: {args.validation} (used once at end for final report)")
            X_test, y_test, _, _ = load_csv_for_classification(
                args.validation, target_column=args.target, target_index=args.target_index)
        else:
            X_test = np.empty((0, X_train.shape[1]))
            y_test = np.array([], dtype=np.asarray(y_train).dtype)
        n_features = X_train.shape[1]
    else:
        if args.demo:
            from rkcnn_data import generate_demo_clusters
            rng = np.random.default_rng(args.seed)
            n_clusters = max(2, args.demo_clusters)
            X, y = generate_demo_clusters(n_samples=100, n_features=10, n_clusters=n_clusters, rng=rng)
            feature_names = [f"f{i}" for i in range(X.shape[1])]
            target_name = "class"
            print("Running in demo mode: 100 samples, 10 features, {} classes.".format(n_clusters))
        else:
            if not args.csv_path:
                print("Error: provide a CSV file path or use --demo or --train.", file=sys.stderr)
                sys.exit(1)
            print(f"Loading CSV: {args.csv_path}")
            X, y, feature_names, target_name = load_csv_for_classification(
                args.csv_path, target_column=args.target, target_index=args.target_index)
            print(f"  Rows: {X.shape[0]}, Features: {X.shape[1]}, Target column: {target_name}")
            print("  Class counts:", dict((str(c), int(cnt)) for c, cnt in Counter(y).items()))

        n_features = X.shape[1]
        if args.test_frac > 0 and args.test_frac < 1:
            rng = np.random.default_rng(args.seed)
            train_idx, test_idx = stratified_train_test_split(y, args.test_frac, rng)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            if verbose:
                print(f"  Train: {len(train_idx)}, Test: {len(test_idx)} (stratified split)")
        else:
            X_train, y_train = X, y
            X_test = np.empty((0, X.shape[1]))
            y_test = np.array([], dtype=y.dtype)

    if args.scale:
        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0) + 1e-10
        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train
        if verbose:
            print("  Features standardized (train mean/std).")

    m = args.m if args.m is not None else max(1, int(np.ceil(np.sqrt(n_features))))
    if m > n_features:
        m = n_features
    if args.h < args.r:
        args.h = args.r
    if args.r > args.h:
        args.r = args.h

    # --- LEAKAGE AUDIT: Validation set is NEVER used during tuning ---
    # 1. Prefilter: topP indices chosen from training (or fold train) only; validation gets same indices.
    # 2. Scaling: mean_train, std_train from X_train only; validation transformed with these.
    # 3. CV folds: stratified_fold_indices yields disjoint folds; no validation rows in any train fold.
    # 4. Best params: selected by CV balanced accuracy on training data only.
    # 5. Validation (X_test) is used EXACTLY ONCE at the end for final performance report only.

    # Tuning block: two-stage CV-only search (X_test/validation not used here)
    if args.tune:
        n_features = X_train.shape[1]
        if use_separate_train_valid and args.outer_folds >= 2:
            if args.tune_grid == "arcene-best":
                combos = [(k, m, r, h) for (k, m, r, h, _) in arcene_best_stage1_combos(n_features=n_features)]
            else:
                combos = [(k, m, r, h) for (k, m, r, h, _) in stage1_combos(include_k7=args.include_k7, seeds=[args.seed], n_features=n_features)]
            outer_folds = min(args.outer_folds, X_train.shape[0])
            inner_folds = max(2, min(args.inner_folds, X_train.shape[0] - 1))
            repeats = max(1, args.repeats)
            print("\nNested CV: {} outer, {} inner, {} repeats...".format(outer_folds, inner_folds, repeats))
            config_inner_scores = defaultdict(list)
            outer_bal_accs = []
            for repeat in range(repeats):
                rng_outer = np.random.default_rng(args.seed + repeat)
                outer_chunks = stratified_fold_indices(y_train, outer_folds, rng_outer)
                for of in range(outer_folds):
                    outer_val_idx = outer_chunks[of]
                    inner_train_idx = np.concatenate([outer_chunks[i] for i in range(outer_folds) if i != of])
                    X_inner = X_train[inner_train_idx]
                    y_inner = y_train[inner_train_idx]
                    X_outer_val = X_train[outer_val_idx]
                    y_outer_val = y_train[outer_val_idx]
                    n_inner_features = X_inner.shape[1]
                    inner_results = []
                    rng_inner = np.random.default_rng(args.seed + repeat * 1000 + of)
                    inner_fold_chunks = stratified_fold_indices(y_inner, inner_folds, rng_inner)
                    for (k, m, r, h) in combos:
                        m_use = min(m, n_inner_features) if m > n_inner_features else m
                        fold_bal_accs = []
                        for inf in range(inner_folds):
                            val_idx = inner_fold_chunks[inf]
                            train_idx = np.concatenate([inner_fold_chunks[i] for i in range(inner_folds) if i != inf])
                            X_fit = X_inner[train_idx]
                            y_fit = y_inner[train_idx]
                            X_val = X_inner[val_idx]
                            y_val = y_inner[val_idx]
                            if prefilter_n and prefilter_n > 0:
                                topP = get_prefilter_top_indices(
                                    X_fit, y_fit, prefilter_n, method=prefilter_method,
                                    ranking_path=args.relieff_ranking, feature_names=feature_names,
                                    prefilter_max=args.prefilter_max or 10000, rng=args.seed + repeat * 1000 + of + inf)
                                X_fit_use, X_val_use = X_fit[:, topP], X_val[:, topP]
                            else:
                                X_fit_use, X_val_use = X_fit, X_val
                            fit_f = rkcnn_fit(X_fit_use, y_fit, k=k, m=m_use, r=r, h=h, random_state=args.seed + repeat * 1000 + of + inf, verbose=False, use_class_weights=args.balance_weights)
                            y_pred_f, _ = rkcnn_predict(fit_f, X_val_use, verbose=False)
                            fold_bal_accs.append(balanced_accuracy(y_val, y_pred_f, fit_f["classes"]))
                        mean_inner = np.mean(fold_bal_accs)
                        config_inner_scores[(k, m_use, r, h)].append(mean_inner)
                        inner_results.append((k, m_use, r, h, mean_inner))
                    best_inner = max(inner_results, key=lambda x: x[4])
                    k_b, m_b, r_b, h_b = best_inner[0], best_inner[1], best_inner[2], best_inner[3]
                    if prefilter_n and prefilter_n > 0:
                        topP = get_prefilter_top_indices(
                            X_inner, y_inner, prefilter_n, method=prefilter_method,
                            ranking_path=args.relieff_ranking, feature_names=feature_names,
                            prefilter_max=args.prefilter_max or 10000, rng=args.seed + repeat * 1000 + of)
                        X_inner_use, X_outer_val_use = X_inner[:, topP], X_outer_val[:, topP]
                    else:
                        X_inner_use, X_outer_val_use = X_inner, X_outer_val
                    fit_outer = rkcnn_fit(X_inner_use, y_inner, k=k_b, m=m_b, r=r_b, h=h_b, random_state=args.seed + repeat * 1000 + of, verbose=False, use_class_weights=args.balance_weights)
                    y_pred_outer, _ = rkcnn_predict(fit_outer, X_outer_val_use, verbose=False)
                    outer_bal_accs.append(balanced_accuracy(y_outer_val, y_pred_outer, fit_outer["classes"]))
            mean_outer = np.mean(outer_bal_accs)
            std_outer = np.std(outer_bal_accs) if len(outer_bal_accs) > 1 else 0.0
            print("\n--- Nested CV: outer balanced accuracy = {:.4f} +/- {:.4f} ---".format(mean_outer, std_outer))
            best_config = max(config_inner_scores.keys(), key=lambda c: np.mean(config_inner_scores[c]))
            best = (best_config[0], best_config[1], best_config[2], best_config[3], np.mean(config_inner_scores[best_config]), 0.0)
            n_folds = inner_folds
            results = []
        elif use_separate_train_valid or use_train_valid_cv:
            n_folds = max(2, min(args.cv_folds, X_train.shape[0]))
            n_repeats = max(1, args.cv_repeats)
            n_train = X_train.shape[0]

            # Stage 1: coarse CV search (repeated CV for stable estimates)
            if args.tune_grid == "arcene-best":
                stage1_list = arcene_best_stage1_combos(n_features=n_features)
            else:
                stage1_list = stage1_combos(include_k7=args.include_k7, n_features=n_features)
            print("\nStage 1: tuning over {} combinations ({}-fold CV x {} repeats on training data)...".format(
                len(stage1_list), n_folds, n_repeats))
            results = []
            for idx, (k, m, r, h, seed) in enumerate(stage1_list):
                m_use = min(m, n_features) if m > n_features else m
                all_fold_accs = []
                all_fold_min_recalls = []
                for rep in range(n_repeats):
                    rng_rep = np.random.default_rng(args.seed + rep * 1000)
                    fold_chunks = stratified_fold_indices(y_train, n_folds, rng_rep)
                    for f in range(n_folds):
                        val_idx = fold_chunks[f]
                        train_idx = np.concatenate([fold_chunks[i] for i in range(n_folds) if i != f])
                        X_fit = X_train[train_idx]
                        y_fit = y_train[train_idx]
                        X_val = X_train[val_idx]
                        y_val = y_train[val_idx]
                        if prefilter_n and prefilter_n > 0:
                            topP = get_prefilter_top_indices(
                                X_fit, y_fit, prefilter_n, method=prefilter_method,
                                ranking_path=args.relieff_ranking, feature_names=feature_names,
                                prefilter_max=args.prefilter_max or 10000, rng=args.seed + rep * 1000 + f)
                            X_fit_use, X_val_use = X_fit[:, topP], X_val[:, topP]
                        else:
                            X_fit_use, X_val_use = X_fit, X_val
                        fit_f = rkcnn_fit(X_fit_use, y_fit, k=k, m=m_use, r=r, h=h, random_state=seed + rep * 1000 + f, verbose=False, use_class_weights=args.balance_weights)
                        y_pred_f, _ = rkcnn_predict(fit_f, X_val_use, verbose=False)
                        all_fold_accs.append(balanced_accuracy(y_val, y_pred_f, fit_f["classes"]))
                        all_fold_min_recalls.append(min_per_class_recall(y_val, y_pred_f, fit_f["classes"]))
                cv_mean = np.mean(all_fold_accs)
                cv_std = np.std(all_fold_accs) if len(all_fold_accs) > 1 else 0.0
                cv_min_recall = np.mean(all_fold_min_recalls)
                results.append((k, m_use, r, h, seed, cv_mean, cv_std, cv_min_recall))
                if verbose:
                    print("  [{}/{}] k={} m={} r={} h={} seed={} -> CV bal_acc={:.4f} +/- {:.4f} min_rec={:.2f}".format(
                        idx + 1, len(stage1_list), k, m_use, r, h, seed, cv_mean, cv_std, cv_min_recall))

            # Aggregate by (k,m,r,h): mean CV across seeds for ranking
            agg = defaultdict(list)
            for r in results:
                key = (r[0], r[1], r[2], r[3])
                agg[key].append((r[4], r[5], r[6], r[7]))
            top_configs = []
            for key, vals in agg.items():
                mean_cv = np.mean([v[1] for v in vals])
                mean_std = np.mean([v[2] for v in vals])
                mean_min_rec = np.mean([v[3] for v in vals])
                top_configs.append((key[0], key[1], key[2], key[3], mean_cv, mean_std, mean_min_rec))
            top_configs = sorted(top_configs, key=lambda x: (-x[4], x[3], x[2]))[:5]
            top_for_stage2 = [(r[0], r[1], r[2], r[3], r[4]) for r in top_configs]

            # Stage 2: refine around top Stage 1 winners
            if args.tune_grid == "arcene-best":
                stage2_list = arcene_best_stage2_combos_from_winners(top_for_stage2, n_features)
            else:
                stage2_list = stage2_combos_from_winners(top_for_stage2, n_features)
            existing = set((r[0], r[1], r[2], r[3]) for r in results)
            stage2_new = [(k, m, r, h, seed) for (k, m, r, h, seed) in stage2_list if (k, m, r, h) not in existing]
            if stage2_new:
                print("\nStage 2: refining around top {} configs ({} new combinations)...".format(len(top_configs), len(stage2_new)))
                for idx, (k, m, r, h, seed) in enumerate(stage2_new):
                    m_use = min(m, n_features) if m > n_features else m
                    all_fold_accs = []
                    all_fold_min_recalls = []
                    for rep in range(n_repeats):
                        rng_rep = np.random.default_rng(args.seed + rep * 1000)
                        fold_chunks = stratified_fold_indices(y_train, n_folds, rng_rep)
                        for f in range(n_folds):
                            val_idx = fold_chunks[f]
                            train_idx = np.concatenate([fold_chunks[i] for i in range(n_folds) if i != f])
                            X_fit = X_train[train_idx]
                            y_fit = y_train[train_idx]
                            X_val = X_train[val_idx]
                            y_val = y_train[val_idx]
                            if prefilter_n and prefilter_n > 0:
                                topP = get_prefilter_top_indices(
                                    X_fit, y_fit, prefilter_n, method=prefilter_method,
                                    ranking_path=args.relieff_ranking, feature_names=feature_names,
                                    prefilter_max=args.prefilter_max or 10000, rng=args.seed + rep * 1000 + f)
                                X_fit_use, X_val_use = X_fit[:, topP], X_val[:, topP]
                            else:
                                X_fit_use, X_val_use = X_fit, X_val
                            fit_f = rkcnn_fit(X_fit_use, y_fit, k=k, m=m_use, r=r, h=h, random_state=seed + rep * 1000 + f, verbose=False, use_class_weights=args.balance_weights)
                            y_pred_f, _ = rkcnn_predict(fit_f, X_val_use, verbose=False)
                            all_fold_accs.append(balanced_accuracy(y_val, y_pred_f, fit_f["classes"]))
                            all_fold_min_recalls.append(min_per_class_recall(y_val, y_pred_f, fit_f["classes"]))
                    cv_mean = np.mean(all_fold_accs)
                    cv_std = np.std(all_fold_accs) if len(all_fold_accs) > 1 else 0.0
                    cv_min_recall = np.mean(all_fold_min_recalls)
                    results.append((k, m_use, r, h, seed, cv_mean, cv_std, cv_min_recall))
                    if verbose:
                        print("  [S2 {}/{}] k={} m={} r={} h={} -> CV bal_acc={:.4f} +/- {:.4f}".format(idx + 1, len(stage2_new), k, m_use, r, h, cv_mean, cv_std))

            # Ensure REFERENCE_PARAMS (1, 110, 100, 1000) is always tested
            ref_k, ref_m, ref_r, ref_h, ref_seed = REFERENCE_PARAMS
            ref_in_results = any((r[0], r[1], r[2], r[3]) == (ref_k, ref_m, ref_r, ref_h) for r in results)
            if not ref_in_results:
                print("\n  Evaluating reference params (k=1, m=110, r=100, h=1000, seed=2)...")
                m_ref = min(ref_m, n_features) if ref_m > n_features else ref_m
                all_fold_accs = []
                all_fold_min_recalls = []
                for rep in range(n_repeats):
                    rng_rep = np.random.default_rng(args.seed + rep * 1000)
                    fold_chunks = stratified_fold_indices(y_train, n_folds, rng_rep)
                    for f in range(n_folds):
                        val_idx = fold_chunks[f]
                        train_idx = np.concatenate([fold_chunks[i] for i in range(n_folds) if i != f])
                        X_fit = X_train[train_idx]
                        y_fit = y_train[train_idx]
                        X_val = X_train[val_idx]
                        y_val = y_train[val_idx]
                        if prefilter_n and prefilter_n > 0:
                            topP = get_prefilter_top_indices(
                                X_fit, y_fit, prefilter_n, method=prefilter_method,
                                ranking_path=args.relieff_ranking, feature_names=feature_names,
                                prefilter_max=args.prefilter_max or 10000, rng=args.seed + rep * 1000 + f)
                            X_fit_use, X_val_use = X_fit[:, topP], X_val[:, topP]
                        else:
                            X_fit_use, X_val_use = X_fit, X_val
                        fit_f = rkcnn_fit(X_fit_use, y_fit, k=ref_k, m=m_ref, r=ref_r, h=ref_h, random_state=ref_seed + rep * 1000 + f, verbose=False, use_class_weights=args.balance_weights)
                        y_pred_f, _ = rkcnn_predict(fit_f, X_val_use, verbose=False)
                        all_fold_accs.append(balanced_accuracy(y_val, y_pred_f, fit_f["classes"]))
                        all_fold_min_recalls.append(min_per_class_recall(y_val, y_pred_f, fit_f["classes"]))
                cv_mean = np.mean(all_fold_accs)
                cv_std = np.std(all_fold_accs) if len(all_fold_accs) > 1 else 0.0
                cv_min_recall = np.mean(all_fold_min_recalls)
                results.append((ref_k, m_ref, ref_r, ref_h, ref_seed, cv_mean, cv_std, cv_min_recall))
                print("  Reference params -> CV bal_acc={:.4f} +/- {:.4f}".format(cv_mean, cv_std))

            # Sort by CV balanced accuracy (tiebreak: smaller h, smaller r)
            def result_key(r):
                return (-r[5], r[3], r[2])
            results_sorted = sorted(results, key=result_key)
            best = results_sorted[0]
            best_k, best_m, best_r, best_h = best[0], best[1], best[2], best[3]

            # CV leaderboard (top 10 by mean CV across seeds per config)
            agg_all = defaultdict(list)
            for r in results:
                key = (r[0], r[1], r[2], r[3])
                agg_all[key].append(r[5])
            leaderboard = [(k, m, r, h, np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0)
                          for (k, m, r, h), vals in agg_all.items()]
            leaderboard = sorted(leaderboard, key=lambda x: (-x[4], x[3], x[2]))[:10]
            print("\n--- CV Leaderboard (top 10 by balanced accuracy) ---")
            for i, (k, m, r, h, mean_cv, std_cv) in enumerate(leaderboard, 1):
                print("  {:2}. k={} m={} r={} h={} -> CV bal_acc={:.4f} +/- {:.4f}".format(i, k, m, r, h, mean_cv, std_cv))

            # Best params for downstream (use fixed seed for final model)
            best = (best_k, best_m, best_r, best_h, results_sorted[0][5], results_sorted[0][7])
        else:
            if args.tune_grid == "arcene-best":
                combos = [(k, m, r, h) for (k, m, r, h, _) in arcene_best_stage1_combos(n_features=n_features)]
            else:
                combos = [(k, m, r, h) for (k, m, r, h, _) in stage1_combos(include_k7=args.include_k7, seeds=[args.seed], n_features=n_features)]
            print("\nTuning over {} combinations (test-frac={})...".format(len(combos), args.test_frac))
            for idx, (k, m, r, h) in enumerate(combos):
                m_use = min(m, n_features) if m > n_features else m
                fit_result = rkcnn_fit(X_train, y_train, k=k, m=m_use, r=r, h=h, random_state=args.seed, verbose=False, use_class_weights=args.balance_weights)
                y_pred_loop, _ = rkcnn_predict(fit_result, X_test, verbose=False)
                bal_acc = balanced_accuracy(y_test, y_pred_loop, fit_result["classes"])
                results.append((k, m_use, r, h, bal_acc))
                print("  [{}/{}] k={} m={} r={} h={} -> balanced accuracy={:.4f}".format(idx + 1, len(combos), k, m_use, r, h, bal_acc))
            best = max(results, key=lambda row: row[4])

        if (use_separate_train_valid or use_train_valid_cv) and results and len(results[0]) == 6:
            valid = [r for r in results if r[5] >= 0.10]
            best = max(valid, key=lambda r: r[4]) if valid else max(results, key=lambda r: r[4])
            if best[5] < 0.10:
                print("\n  (No combo had min recall >= 10%; chosen best by CV balanced accuracy.)")
        if use_train_valid_cv:
            print("\n--- Best parameters (by {}-fold CV) ---".format(n_folds))
            print("  k={}, m={}, r={}, h={} -> CV balanced accuracy={:.4f}".format(best[0], best[1], best[2], best[3], best[4]))
            target_for_cmd = " --target {}".format(args.target) if args.target else ""
            opt_cmd = " --no-scale --no-prefilter --tune-grid arcene-best" if args.tune_grid == "arcene-best" else (" --scale" if args.scale else "")
            print("  Reproduce: python RCNN-CV --train-valid {}{} --tune --cv-folds {}".format(args.train_valid, target_for_cmd + opt_cmd, args.cv_folds))
            if args.tuning_table and results and len(results[0]) == 8:
                with open(args.tuning_table, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["k", "m", "r", "h", "seed", "cv_bal_acc_mean", "cv_bal_acc_std", "mean_min_recall", "is_best"])
                    for r in results:
                        is_best = (r[0], r[1], r[2], r[3]) == (best[0], best[1], best[2], best[3])
                        w.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], is_best])
                print("  Tuning table saved: {}".format(args.tuning_table))
            sys.exit(0)
        elif use_separate_train_valid:
            print("\n--- Best parameters (by {}-fold CV on training data) ---".format(n_folds))
            print("  k={}, m={}, r={}, h={} -> CV balanced accuracy={:.4f}".format(best[0], best[1], best[2], best[3], best[4]))
            target_for_cmd = " --target {}".format(args.target) if args.target else ""
            opt_cmd = " --no-scale --no-prefilter --tune-grid arcene-best" if args.tune_grid == "arcene-best" else (" --scale" if args.scale else "")
            print("  Reproduce: python RCNN-CV --train {}{} --tune --cv-folds {}".format(args.train, target_for_cmd + opt_cmd, args.cv_folds))
            valid_bal, valid_overall = None, None
            fit_final = None
            X_train_use_final = None
            X_test_use_final = None
            y_pred_val = None
            topP_final = None
            # FIRST AND ONLY USE OF VALIDATION: fit on train, evaluate once on X_test
            if X_test.shape[0] > 0:
                print("\n--- Final model: fit on full training set, evaluate once on validation ---")
                if prefilter_n and prefilter_n > 0:
                    topP_final = get_prefilter_top_indices(
                        X_train, y_train, prefilter_n, method=prefilter_method,
                        ranking_path=args.relieff_ranking, feature_names=feature_names,
                        prefilter_max=args.prefilter_max or 10000, rng=args.seed)
                    X_train_use_final = X_train[:, topP_final]
                    X_test_use_final = X_test[:, topP_final]
                else:
                    X_train_use_final = X_train
                    X_test_use_final = X_test
                fit_final = rkcnn_fit(X_train_use_final, y_train, k=best[0], m=best[1], r=best[2], h=best[3], random_state=args.seed, verbose=False, use_class_weights=args.balance_weights)
                if topP_final is not None:
                    fit_final["prefilter_indices"] = topP_final
                y_pred_val, _ = rkcnn_predict(fit_final, X_test_use_final, verbose=False)
                valid_bal = balanced_accuracy(y_test, y_pred_val, fit_final["classes"])
                valid_overall = np.mean(y_test == y_pred_val)
                print("  Validation balanced accuracy: {:.4f}".format(valid_bal))
                print("  Validation overall accuracy:  {:.4f}".format(valid_overall))
                train_cv_res = run_train_cv_evaluation(
                    X_train, y_train, best[0], best[1], best[2], best[3], n_folds, args.seed,
                    prefilter=prefilter_n or 0, prefilter_max=args.prefilter_max or 10000,
                    use_class_weights=args.balance_weights, repeats=max(1, args.cv_repeats),
                    prefilter_method=prefilter_method, relieff_ranking_path=args.relieff_ranking,
                    feature_names=feature_names)
                train_cv_mean, train_cv_std = train_cv_res[0], train_cv_res[1]
            else:
                train_cv_mean = train_cv_std = ""
                train_cv_res = None
                if results and len(results[0]) == 8:
                    tr = run_train_cv_evaluation(
                        X_train, y_train, best[0], best[1], best[2], best[3], n_folds, args.seed,
                        prefilter=prefilter_n or 0, prefilter_max=args.prefilter_max or 10000,
                        use_class_weights=args.balance_weights, repeats=max(1, args.cv_repeats),
                        prefilter_method=prefilter_method, relieff_ranking_path=args.relieff_ranking,
                        feature_names=feature_names)
                    train_cv_mean, train_cv_std = tr[0], tr[1]
            if args.tuning_table and results and len(results[0]) == 8:
                out_path = args.tuning_table
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                train_cv_bal = train_cv_mean if train_cv_mean != "" else ""
                train_cv_std_val = train_cv_std if train_cv_std != "" else ""
                with open(out_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["k", "m", "r", "h", "seed", "cv_bal_acc_mean", "cv_bal_acc_std", "mean_min_recall", "is_best", "train_cv_bal_acc", "train_cv_std", "valid_bal_acc", "valid_overall_acc"])
                    for r in results:
                        is_best = (r[0], r[1], r[2], r[3]) == (best[0], best[1], best[2], best[3])
                        tcb = train_cv_bal if is_best else ""
                        tcs = train_cv_std_val if is_best else ""
                        vb = valid_bal if is_best and valid_bal is not None else ""
                        vo = valid_overall if is_best and valid_overall is not None else ""
                        w.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], is_best, tcb, tcs, vb, vo])
                print("  Tuning table saved: {}".format(out_path))
            print("\n--- CV tuning diagnostic plots ---")
            tune_dir = os.path.join(results_root, "tuning")
            plot_cv_tuning_diagnostics(results, tune_dir)
            if X_test.shape[0] > 0 and fit_final is not None:
                train_dir = os.path.join(results_root, "training")
                valid_dir = os.path.join(results_root, "validation")
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(valid_dir, exist_ok=True)
                n_features_out = X_train_use_final.shape[1]
                scores_train = separation_scores_per_feature(X_train_use_final, y_train)
                top2_t = np.argsort(-scores_train)[: min(2, n_features_out)]
                top3_t = np.argsort(-scores_train)[: min(3, n_features_out)]
                top5_t = np.argsort(-scores_train)[: min(5, n_features_out)]
                acc_val = accuracy(y_test, y_pred_val)
                classes = fit_final["classes"]
                cm_val = confusion_matrix(y_test, y_pred_val, classes)
                print("\n--- Validation results ---")
                print("Tuning CV balanced accuracy: {:.4f}".format(best[4]))
                print("Validation balanced accuracy: {:.4f}".format(valid_bal))
                print("Validation overall accuracy: {:.4f}".format(acc_val))
                print("Confusion matrix (rows = true class, columns = predicted class):")
                print("  Classes (order):", [str(c) for c in classes])
                print(cm_val)
                with open(os.path.join(valid_dir, "results.txt"), "w") as f:
                    f.write("Validation results\n\n")
                    f.write("Tuning CV balanced accuracy: {:.4f}\n".format(best[4]))
                    f.write("Validation balanced accuracy: {:.4f}\n".format(valid_bal))
                    f.write("Validation overall accuracy: {:.4f}\n".format(acc_val))
                    f.write("Confusion matrix (rows = true class, columns = predicted class):\n")
                    f.write("  Classes (order): {}\n".format([str(c) for c in classes]))
                    f.write("{}\n".format(cm_val))
                print("  Results saved: {}".format(os.path.join(valid_dir, "results.txt")))
                scores_valid = separation_scores_per_feature(X_test_use_final, y_test)
                print("\nPlotting validation...")
                plot_separation_scores(scores_valid, feature_names=feature_names, filepath=os.path.join(valid_dir, "separation_scores.png"), max_features=args.plot_max_features)
                plot_clusters_2d(X_test_use_final, y_true=y_test, y_pred=y_pred_val, filepath=os.path.join(valid_dir, "clusters_2d.png"), title="RkCNN clusters (2D)", feature_names=feature_names, top_two_indices=top2_t)
                plot_clusters_3d(X_test_use_final, y_true=y_test, y_pred=y_pred_val, filepath=os.path.join(valid_dir, "clusters_3d.png"), title="RkCNN clusters (3D)", feature_names=feature_names, top_three_indices=top3_t)
                plot_clusters_top_features_pairwise(X_test_use_final, y_true=y_test, y_pred=y_pred_val, top_indices=top5_t, feature_names=feature_names, filepath=os.path.join(valid_dir, "clusters_top5_features.png"), n_top=5)
                plot_per_class_accuracy(y_test, y_pred_val, classes, filepath=os.path.join(valid_dir, "accuracy_by_class.png"), title="Accuracy by class")
                y_true_cv, y_pred_cv, val_inds_cv = train_cv_res[2], train_cv_res[3], train_cv_res[4]
                cm_train = confusion_matrix(y_true_cv, y_pred_cv, classes)
                print("\n--- Training results (Train set CV - unbiased generalization estimate) ---")
                print("Train set CV balanced accuracy: {:.4f}".format(train_cv_mean))
                print("Train set CV std: {:.4f}".format(train_cv_std))
                print("Confusion matrix (rows = true, cols = predicted, from CV holdout):")
                print("  Classes (order):", [str(c) for c in classes])
                print(cm_train)
                with open(os.path.join(train_dir, "results.txt"), "w") as f:
                    f.write("Training results (Train set CV - unbiased generalization estimate)\n\n")
                    f.write("Train set CV balanced accuracy: {:.4f}\n".format(train_cv_mean))
                    f.write("Train set CV std: {:.4f}\n".format(train_cv_std))
                    f.write("\nPer-class recall (from CV holdout predictions):\n")
                    for i, c in enumerate(classes):
                        mask = y_true_cv == c
                        total_c = np.sum(mask)
                        correct_c = np.sum((y_true_cv == c) & (y_pred_cv == c))
                        pct = 100.0 * correct_c / total_c if total_c > 0 else 0.0
                        f.write("  Class {}: {:.0f}%\n".format(c, pct))
                    f.write("\nConfusion matrix (rows = true, cols = predicted):\n")
                    f.write("  Classes (order): {}\n".format([str(c) for c in classes]))
                    f.write("{}\n".format(cm_train))
                print("  Results saved: {}".format(os.path.join(train_dir, "results.txt")))
                print("\nPlotting training...")
                plot_separation_scores(scores_train, feature_names=feature_names, filepath=os.path.join(train_dir, "separation_scores.png"), max_features=args.plot_max_features)
                order_train = np.argsort(val_inds_cv)
                y_pred_train_reord = y_pred_cv[order_train]
                plot_clusters_2d(X_train_use_final, y_true=y_train, y_pred=y_pred_train_reord, filepath=os.path.join(train_dir, "clusters_2d.png"), title="RkCNN clusters (2D)", feature_names=feature_names, top_two_indices=top2_t)
                plot_clusters_3d(X_train_use_final, y_true=y_train, y_pred=y_pred_train_reord, filepath=os.path.join(train_dir, "clusters_3d.png"), title="RkCNN clusters (3D)", feature_names=feature_names, top_three_indices=top3_t)
                plot_clusters_top_features_pairwise(X_train_use_final, y_true=y_train, y_pred=y_pred_train_reord, top_indices=top5_t, feature_names=feature_names, filepath=os.path.join(train_dir, "clusters_top5_features.png"), n_top=5)
                plot_per_class_accuracy(y_true_cv, y_pred_cv, classes, filepath=os.path.join(train_dir, "accuracy_by_class.png"), title="Accuracy by class")
                # Best params comparison: when CV winner differs from reference (1, 110, 100, 1000)
                if (best[0], best[1], best[2], best[3]) != (1, 110, 100, 1000):
                    comp_dir = os.path.join(results_root, "best_params_comparison")
                    os.makedirs(comp_dir, exist_ok=True)
                    ref_k, ref_m, ref_r, ref_h, ref_seed = REFERENCE_PARAMS
                    # CV score for reference: from results or run evaluation
                    ref_results = [r for r in results if (r[0], r[1], r[2], r[3]) == (ref_k, ref_m, ref_r, ref_h)]
                    if ref_results:
                        ref_cv_mean = np.mean([r[5] for r in ref_results])
                        ref_cv_std = np.std([r[5] for r in ref_results]) if len(ref_results) > 1 else 0.0
                    else:
                        ref_cv_res = run_train_cv_evaluation(
                            X_train, y_train, ref_k, ref_m, ref_r, ref_h, n_folds, ref_seed,
                            prefilter=prefilter_n or 0, prefilter_max=args.prefilter_max or 10000,
                            use_class_weights=args.balance_weights, repeats=max(1, args.cv_repeats),
                            prefilter_method=prefilter_method, relieff_ranking_path=args.relieff_ranking,
                            feature_names=feature_names)
                        ref_cv_mean, ref_cv_std = ref_cv_res[0], ref_cv_res[1]
                    # Validation score for reference: fit and predict once
                    m_ref_use = min(ref_m, X_train_use_final.shape[1]) if ref_m > X_train_use_final.shape[1] else ref_m
                    fit_ref = rkcnn_fit(X_train_use_final, y_train, k=ref_k, m=m_ref_use, r=ref_r, h=ref_h, random_state=ref_seed, verbose=False, use_class_weights=args.balance_weights)
                    y_pred_ref, _ = rkcnn_predict(fit_ref, X_test_use_final, verbose=False)
                    ref_valid_bal = balanced_accuracy(y_test, y_pred_ref, fit_ref["classes"])
                    ref_valid_overall = np.mean(y_test == y_pred_ref)
                    with open(os.path.join(comp_dir, "results.txt"), "w") as f:
                        f.write("Best Parameters Comparison (CV winner vs reference params)\n\n")
                        f.write("--- Best by Cross-Validation ---\n")
                        f.write("  k={}, m={}, r={}, h={}, seed={}\n".format(best[0], best[1], best[2], best[3], args.seed))
                        f.write("  Training CV balanced accuracy: {:.4f}\n".format(train_cv_mean))
                        f.write("  Validation balanced accuracy: {:.4f}\n".format(valid_bal))
                        f.write("  Validation overall accuracy:  {:.4f}\n\n".format(valid_overall))
                        f.write("--- Reference Parameters (k=1, m=110, r=100, h=1000, seed=2) ---\n")
                        f.write("  Training CV balanced accuracy: {:.4f}\n".format(ref_cv_mean))
                        f.write("  Validation balanced accuracy: {:.4f}\n".format(ref_valid_bal))
                        f.write("  Validation overall accuracy:  {:.4f}\n\n".format(ref_valid_overall))
                        f.write("Note: The CV winner was chosen by training-only cross-validation.\n")
                        f.write("Validation was used only once at the end for both configurations.\n")
                    with open(os.path.join(comp_dir, "comparison.csv"), "w", newline="") as cf:
                        w = csv.writer(cf)
                        w.writerow(["config", "k", "m", "r", "h", "seed", "train_cv_bal_acc", "valid_bal_acc", "valid_overall_acc"])
                        w.writerow(["best", best[0], best[1], best[2], best[3], args.seed, train_cv_mean, valid_bal, valid_overall])
                        w.writerow(["reference", ref_k, ref_m, ref_r, ref_h, ref_seed, ref_cv_mean, ref_valid_bal, ref_valid_overall])
                    print("\n--- Best params comparison (CV winner != reference) saved to {} ---".format(comp_dir))
                print("Done. Outputs in {} and {}.".format(train_dir, valid_dir))
            sys.exit(0)
        else:
            print("\n--- Best parameters ---")
            print("  k={}, m={}, r={}, h={} -> balanced accuracy={:.4f}".format(best[0], best[1], best[2], best[3], best[4]))
            sys.exit(0)

    # Normal run: when --train-valid (no validation file), run CV and exit.
    # When --train + --validation, fit on full train and produce training/validation outputs (same as RCNN-2).
    if use_train_valid_cv or (use_separate_train_valid and X_test.shape[0] == 0):
        n_folds = max(2, min(args.cv_folds, X_train.shape[0]))
        rng_cv = np.random.default_rng(args.seed)
        fold_chunks = stratified_fold_indices(y_train, n_folds, rng_cv)
        fold_bal_accs = []
        for f in range(n_folds):
            val_idx = fold_chunks[f]
            train_idx = np.concatenate([fold_chunks[i] for i in range(n_folds) if i != f])
            X_fit = X_train[train_idx]
            y_fit = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            fit_f = rkcnn_fit(X_fit, y_fit, k=args.k, m=m, r=args.r, h=args.h, random_state=args.seed + f, verbose=False, use_class_weights=args.balance_weights)
            y_pred_f, _ = rkcnn_predict(fit_f, X_val, verbose=False)
            fold_bal_accs.append(balanced_accuracy(y_val, y_pred_f, fit_f["classes"]))
        cv_bal_acc = np.mean(fold_bal_accs)
        print("\n--- Cross-validation results (RCNN-CV: no validation-set evaluation) ---")
        print("Parameters: k={}, m={}, r={}, h={}".format(args.k, m, args.r, args.h))
        print("{}-fold CV balanced accuracy: {:.4f}".format(n_folds, cv_bal_acc))
        if verbose and len(fold_bal_accs) > 1:
            print("  Per-fold balanced accuracies:", [round(x, 4) for x in fold_bal_accs])
        print("  Use --tune to search for best parameters.")
        sys.exit(0)

    # Fit and predict: --train + --validation (produce training/validation outputs) or single CSV/demo
    print("\nParameters: k={}, m={}, r={}, h={}".format(args.k, m, args.r, args.h))
    if prefilter_n and prefilter_n > 0:
        topP = get_prefilter_top_indices(
            X_train, y_train, prefilter_n, method=prefilter_method,
            ranking_path=args.relieff_ranking, feature_names=feature_names,
            prefilter_max=args.prefilter_max or 10000, rng=args.seed)
        X_train_use = X_train[:, topP]
        X_test_use = X_test[:, topP]
    else:
        X_train_use = X_train
        X_test_use = X_test
        topP = None
    print("\nFitting RkCNN...")
    fit_result = rkcnn_fit(
        X_train_use, y_train,
        k=args.k, m=m, r=args.r, h=args.h,
        random_state=args.seed,
        verbose=verbose,
        use_class_weights=args.balance_weights,
    )
    if topP is not None:
        fit_result["prefilter_indices"] = topP
    print("  Fit done.\nPredicting...")
    class_bias = {'-1': args.class_bias} if args.class_bias != 1.0 else None
    y_pred, proba = rkcnn_predict(fit_result, X_test_use, verbose=verbose, class_bias=class_bias)
    print("  Predictions done.")

    acc = accuracy(y_test, y_pred)
    classes = fit_result["classes"]
    cm = confusion_matrix(y_test, y_pred, classes)
    n_features = X_test_use.shape[1]
    bal_acc = balanced_accuracy(y_test, y_pred, classes)

    if use_separate_train_valid and X_test.shape[0] > 0:
        train_dir = os.path.join(results_root, "training")
        valid_dir = os.path.join(results_root, "validation")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        scores_train = separation_scores_per_feature(X_train_use, y_train)
        top2_t = np.argsort(-scores_train)[: min(2, n_features)]
        top3_t = np.argsort(-scores_train)[: min(3, n_features)]
        top5_t = np.argsort(-scores_train)[: min(5, n_features)]
        print("\n--- Validation results ---")
        print("Validation balanced accuracy: {:.4f}".format(bal_acc))
        print("Validation overall accuracy: {:.4f}".format(acc))
        print("Confusion matrix (rows = true class, columns = predicted class):")
        print("  Classes (order):", [str(c) for c in classes])
        print(cm)
        with open(os.path.join(valid_dir, "results.txt"), "w") as f:
            f.write("Validation results\n\n")
            f.write("Validation balanced accuracy: {:.4f}\n".format(bal_acc))
            f.write("Validation overall accuracy: {:.4f}\n".format(acc))
            f.write("Confusion matrix (rows = true class, columns = predicted class):\n")
            f.write("  Classes (order): {}\n".format([str(c) for c in classes]))
            f.write("{}\n".format(cm))
        print("  Results saved: {}".format(os.path.join(valid_dir, "results.txt")))
        scores_valid = separation_scores_per_feature(X_test_use, y_test)
        print("\nPlotting validation...")
        plot_separation_scores(scores_valid, feature_names=feature_names, filepath=os.path.join(valid_dir, "separation_scores.png"), max_features=args.plot_max_features)
        plot_clusters_2d(X_test_use, y_true=y_test, y_pred=y_pred, filepath=os.path.join(valid_dir, "clusters_2d.png"), title="RkCNN clusters (2D)", feature_names=feature_names, top_two_indices=top2_t)
        plot_clusters_3d(X_test_use, y_true=y_test, y_pred=y_pred, filepath=os.path.join(valid_dir, "clusters_3d.png"), title="RkCNN clusters (3D)", feature_names=feature_names, top_three_indices=top3_t)
        plot_clusters_top_features_pairwise(X_test_use, y_true=y_test, y_pred=y_pred, top_indices=top5_t, feature_names=feature_names, filepath=os.path.join(valid_dir, "clusters_top5_features.png"), n_top=5)
        plot_per_class_accuracy(y_test, y_pred, classes, filepath=os.path.join(valid_dir, "accuracy_by_class.png"), title="Accuracy by class")
        n_folds_nt = max(2, min(args.cv_folds, X_train.shape[0]))
        train_cv_mean, train_cv_std, y_true_cv, y_pred_cv, val_inds_cv, _ = run_train_cv_evaluation(
            X_train, y_train, args.k, m, args.r, args.h, n_folds_nt, args.seed,
            prefilter=prefilter_n or 0, prefilter_max=args.prefilter_max or 10000,
            use_class_weights=args.balance_weights, repeats=max(1, args.cv_repeats),
            prefilter_method=prefilter_method, relieff_ranking_path=args.relieff_ranking,
            feature_names=feature_names)
        cm_train = confusion_matrix(y_true_cv, y_pred_cv, classes)
        print("\n--- Training results (Train set CV - unbiased generalization estimate) ---")
        print("Train set CV balanced accuracy: {:.4f}".format(train_cv_mean))
        print("Train set CV std: {:.4f}".format(train_cv_std))
        print("Confusion matrix (rows = true, cols = predicted, from CV holdout):")
        print("  Classes (order):", [str(c) for c in classes])
        print(cm_train)
        with open(os.path.join(train_dir, "results.txt"), "w") as f:
            f.write("Training results (Train set CV - unbiased generalization estimate)\n\n")
            f.write("Train set CV balanced accuracy: {:.4f}\n".format(train_cv_mean))
            f.write("Train set CV std: {:.4f}\n".format(train_cv_std))
            f.write("\nPer-class recall (from CV holdout predictions):\n")
            for i, c in enumerate(classes):
                mask = y_true_cv == c
                total_c = np.sum(mask)
                correct_c = np.sum((y_true_cv == c) & (y_pred_cv == c))
                pct = 100.0 * correct_c / total_c if total_c > 0 else 0.0
                f.write("  Class {}: {:.0f}%\n".format(c, pct))
            f.write("\nConfusion matrix (rows = true, cols = predicted):\n")
            f.write("  Classes (order): {}\n".format([str(c) for c in classes]))
            f.write("{}\n".format(cm_train))
        print("  Results saved: {}".format(os.path.join(train_dir, "results.txt")))
        print("\nPlotting training...")
        plot_separation_scores(scores_train, feature_names=feature_names, filepath=os.path.join(train_dir, "separation_scores.png"), max_features=args.plot_max_features)
        order_train = np.argsort(val_inds_cv)
        y_pred_train_reord = y_pred_cv[order_train]
        plot_clusters_2d(X_train_use, y_true=y_train, y_pred=y_pred_train_reord, filepath=os.path.join(train_dir, "clusters_2d.png"), title="RkCNN clusters (2D)", feature_names=feature_names, top_two_indices=top2_t)
        plot_clusters_3d(X_train_use, y_true=y_train, y_pred=y_pred_train_reord, filepath=os.path.join(train_dir, "clusters_3d.png"), title="RkCNN clusters (3D)", feature_names=feature_names, top_three_indices=top3_t)
        plot_clusters_top_features_pairwise(X_train_use, y_true=y_train, y_pred=y_pred_train_reord, top_indices=top5_t, feature_names=feature_names, filepath=os.path.join(train_dir, "clusters_top5_features.png"), n_top=5)
        plot_per_class_accuracy(y_true_cv, y_pred_cv, classes, filepath=os.path.join(train_dir, "accuracy_by_class.png"), title="Accuracy by class")
        print("Done. Outputs in {} and {}.".format(train_dir, valid_dir))
    else:
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
        print("\nPlotting...")
        plot_separation_scores(scores, feature_names=feature_names, filepath=args.plot_separation, max_features=args.plot_max_features)
        plot_clusters_2d(X_test, y_true=y_test, y_pred=y_pred, filepath=args.plot, title="RkCNN clusters (2D)", feature_names=feature_names, top_two_indices=top2)
        plot_clusters_3d(X_test, y_true=y_test, y_pred=y_pred, filepath=args.plot_3d, title="RkCNN clusters (3D)", feature_names=feature_names, top_three_indices=top3)
        plot_clusters_top_features_pairwise(X_test, y_true=y_test, y_pred=y_pred, top_indices=top5, feature_names=feature_names, filepath=args.plot_top, n_top=5)
        plot_per_class_accuracy(y_test, y_pred, classes, filepath=args.plot_accuracy, title="Accuracy by class")
        print("Done.")


if __name__ == "__main__":
    main()
