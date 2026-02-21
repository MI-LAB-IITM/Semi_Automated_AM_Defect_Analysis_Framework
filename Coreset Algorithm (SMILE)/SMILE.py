"""
SMILE : Sampling via Maximin LHS from Embeddings

This implementation selects k representative samples per cluster from a 2D embedding space (e.g., t-SNE, UMAP).

For each cluster:
    1) Attempt strict Latin Hypercube Sampling (one per row and column)
    2) If strict feasibility fails(e.g. insufficient bin coverage or no valid row–column permutation),
    fall back to a relaxed grid-based maximin selection.

The goal is to balance coverage (LHS) with diversity (maximin criterion) across clusters 
=> Finally selected points across ensures global coverage.

Designed primarily for small-k, data-scarce scenarios.

"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from itertools import combinations, permutations
from collections import Counter


class SMILE:
    def __init__(self, n_clusters=6, k=4, random_state=42, verbose=False):
        self.n_clusters = n_clusters
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.labels_ = None
        self.cluster_lhs_indices_ = {}


    def fit_clusters(self, X): #Run K-means clustering for clusters
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto"
        )
        self.labels_ = kmeans.fit_predict(X)
        return self.labels_


    @staticmethod
    def _maximin_tuple(X, decimals=10): #Maximin
        D = pdist(X)
        rounded = np.round(D, decimals=decimals)
        counter = Counter(rounded)

        d_sorted = np.sort(list(counter.keys()))
        J = np.array([counter[d] for d in d_sorted])

        return tuple(np.ravel(np.column_stack([d_sorted, -J])))

    @staticmethod
    def _compare_maximin(t1, t2): #comapare
        for a, b in zip(t1, t2):
            if a > b:
                return 1
            elif a < b:
                return -1
        return 0

    @staticmethod
    def _normalize(X): 
        denom = (X.max(axis=0) - X.min(axis=0)) + 1e-12 #Normalize clusters
        return (X - X.min(axis=0)) / denom


    def _strict_lhs_maximin(self, X): #strict lhs_maximin to ensure coverage and diversity
        k = self.k
        Xn = self._normalize(X)

        bins = [[[] for _ in range(k)] for _ in range(k)]

        for i, (x, y) in enumerate(Xn):
            bx = min(int(x * k), k - 1)
            by = min(int(y * k), k - 1)
            bins[bx][by].append(i)

        best_tuple = None
        best_subset = None
        strict_feasible = False

    # Strict LHS requires exactly one bin per row and per column
    # This is equivalent to searching for a valid permutation of columns
    # If at least one permutation is feasible, we evaluate them using maximin

        for perm in permutations(range(k)):
            candidates = []
            feasible = True

            for row in range(k):
                col = perm[row]
                if not bins[row][col]:
                    feasible = False
                    break
                candidates.append(bins[row][col][0])

            if not feasible:
                continue

            strict_feasible = True
            subset = Xn[candidates]
            spacing = self._maximin_tuple(subset)

            if best_tuple is None or \
               self._compare_maximin(spacing, best_tuple) > 0:
                best_tuple = spacing
                best_subset = candidates

        if strict_feasible:
            if self.verbose:
                print("Strict LHS used.")
            return list(best_subset)

        return None

    def _relaxed_lhs_maximin(self, X): #if lhs_maximin is strictly not possible (e.g. insufficient bin coverage or no valid row–column permutation)
        k = self.k
        Xn = self._normalize(X)

        bins = [[[] for _ in range(k)] for _ in range(k)]

        for i, (x, y) in enumerate(Xn):
            bx = min(int(x * k), k - 1)
            by = min(int(y * k), k - 1)
            bins[bx][by].append(i)

        candidates = [
            bins[i][j][0]
            for i in range(k)
            for j in range(k)
            if bins[i][j]
        ]

        if len(candidates) < k:
            return None

        best_tuple = None
        best_subset = None
    # In the relaxed version, we allow multiple selections from the same row or column. We simply take one representative per non-empty bin and optimize spacing.


        for combo in combinations(candidates, k):
            subset = Xn[list(combo)]
            spacing = self._maximin_tuple(subset)

            if best_tuple is None or \
               self._compare_maximin(spacing, best_tuple) > 0:
                best_tuple = spacing
                best_subset = combo

        if self.verbose:
            print("Relaxed LHS used.")

        return list(best_subset)

    def fit(self, X): #Fitting
        
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a NumPy array.")

        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array of shape (n_samples, 2).")

        if X.shape[1] != 2:
            raise ValueError(f"Received input with dimension {X.shape[1]}.")

        if self.labels_ is None:
            self.fit_clusters(X)

        self.cluster_lhs_indices_ = {}

        for cid in range(self.n_clusters):
            cluster_pts = X[self.labels_ == cid]
            cluster_idx = np.where(self.labels_ == cid)[0]

            if cluster_pts.shape[0] < self.k:
                continue

            #Try strict first
            selected_local = self._strict_lhs_maximin(cluster_pts) # Prefer strict LHS when possible, as it guarantees coverage and diversity

            #If strict fails use relaxed
            if selected_local is None:
                selected_local = self._relaxed_lhs_maximin(cluster_pts)

            if selected_local is not None:
                self.cluster_lhs_indices_[cid] = \
                    cluster_idx[selected_local]

        return self.cluster_lhs_indices_

    def get_selected_indices(self): #Get the selected indices
        return self.cluster_lhs_indices_


