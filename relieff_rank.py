#!/usr/bin/env python3
"""
ReliefF feature ranking — ranks all features by predictive power.

Implements ReliefF per Robnik-Sikonja & Kononenko (2003). Uses Manhattan distance
for neighbor search and diff as in Eq. 2 for continuous features.
"""

import argparse
import numpy as np
import pandas as pd


def load_data(csv_path, target_col="class"):
    """Load CSV, return X (n, a), y (n,), feature_names (list of a names)."""
    df = pd.read_csv(csv_path)
    y = df[target_col].astype(str).values
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
    valid = ~(np.isnan(X).any(axis=1))
    X, y = X[valid], y[valid]
    return X, y, feature_cols


def precompute_feature_range(X):
    """Return min_a, max_a each shape (a,)."""
    min_a = X.min(axis=0)
    max_a = X.max(axis=0)
    return min_a, max_a


def diff_continuous(X, feat_idx, i1, i2, min_a, max_a, eps=1e-10):
    """diff(A, I1, I2) per Robnik-Sikonja Eq. 2."""
    val1, val2 = X[i1, feat_idx], X[i2, feat_idx]
    denom = max_a[feat_idx] - min_a[feat_idx] + eps
    return np.abs(val1 - val2) / denom


def distances_from_instance(X, r, min_a, max_a, eps=1e-10):
    """Return (n,) array: distance from instance r to each instance j. Uses Manhattan = sum of diff."""
    d = np.abs(X[r : r + 1] - X) / (max_a - min_a + eps)
    return np.sum(d, axis=1)


def find_k_nearest(X, r, k, mask, min_a, max_a):
    """mask: True for candidates. Exclude r. Return up to k nearest indices."""
    mask = mask.copy()
    mask[r] = False
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int)
    dists = distances_from_instance(X, r, min_a, max_a)
    dists_valid = dists[candidates]
    order = np.argsort(dists_valid)
    k_use = min(k, len(candidates))
    return candidates[order[:k_use]]


def compute_prior(y, prior_type="empirical"):
    """Return dict: class_label -> P(class)."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    if prior_type == "uniform":
        return {str(c): 1.0 / len(classes) for c in classes}
    return {str(c): float(counts[i]) / n for i, c in enumerate(classes)}


def relieff_weights(X, y, k=10, m=None, prior_type="empirical", rng=None):
    """X: (n, a), y: (n,) class labels. Return W: (a,) feature weights."""
    n, a = X.shape
    m = m if m is not None else n
    if rng is None:
        rng = np.random.default_rng(42)

    min_a, max_a = precompute_feature_range(X)
    P = compute_prior(y, prior_type)
    classes = np.unique(y)
    W = np.zeros(a)

    indices = rng.permutation(n)[:m]

    for r in indices:
        y_r = str(y[r])
        # Hits: same class
        hit_mask = (y == y_r)
        hits = find_k_nearest(X, r, k, hit_mask, min_a, max_a)
        k_hits = max(1, len(hits))

        other_classes = [c for c in classes if str(c) != y_r]
        denom_miss = 1.0 - P.get(y_r, 0)
        if denom_miss <= 0:
            continue

        # Precompute misses for each other class (once per r)
        misses_per_class = {}
        for c in other_classes:
            miss_mask = (y == str(c))
            misses_per_class[c] = find_k_nearest(X, r, k, miss_mask, min_a, max_a)

        for feat in range(a):
            # Hit term: penalize
            hit_sum = sum(diff_continuous(X, feat, r, h, min_a, max_a) for h in hits)
            W[feat] -= hit_sum / (m * k_hits)

            # Miss term: reward
            miss_sum = 0.0
            for c in other_classes:
                misses = misses_per_class[c]
                m_sum = sum(diff_continuous(X, feat, r, j, min_a, max_a) for j in misses)
                weight_c = P[str(c)] / denom_miss
                k_miss = max(1, len(misses))
                miss_sum += weight_c * (m_sum / k_miss)
            W[feat] += miss_sum / m

    return W


def main():
    ap = argparse.ArgumentParser(description="ReliefF feature ranking")
    ap.add_argument("--input", default="arcene/arcene_train.csv")
    ap.add_argument("--target", default="class")
    ap.add_argument("--output", default="arcene/relieff_ranking.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--prior", default="empirical", choices=["empirical", "uniform"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, feature_names = load_data(args.input, args.target)
    print("Loaded: {} samples, {} features".format(X.shape[0], X.shape[1]))
    rng = np.random.default_rng(args.seed)
    W = relieff_weights(X, y, k=args.k, m=args.updates, prior_type=args.prior, rng=rng)

    order = np.argsort(-W)
    df = pd.DataFrame(
        [(i + 1, feature_names[order[i]], float(W[order[i]])) for i in range(len(order))],
        columns=["rank", "feature", "weight"],
    )
    df.to_csv(args.output, index=False)
    print("Top 20:\n", df.head(20))
    print("Bottom 5:\n", df.tail(5))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
