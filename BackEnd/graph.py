# Sparse graph and Fermat-geometry construction for the paper experiments.
#
# The module separates three graph families: the released Euclidean PWLL graph,
# the exact sparse-graph Fermat metric used by FALL, and the Landmark-MDS
# approximation used to precompute A-FALL candidate graphs.

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class DistanceEdgeCache:
    # Sparse directed edge distances plus their reference bandwidth.

    n_points: int
    rows: np.ndarray
    columns: np.ndarray
    distances: np.ndarray
    base_epsilon: float


def knn_adjacency(X, k, include_self=False):
    # Build a union-symmetrized sparse Euclidean kNN distance graph.

    X = np.asarray(X, dtype=float)
    n_points = X.shape[0]
    if n_points < 2:
        raise ValueError("Need at least two points.")
    if k <= 0:
        raise ValueError("k must be positive.")

    n_neighbors = min(int(k) + 1, n_points)
    knn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_jobs=-1,
    )
    knn_model.fit(X)
    distances, indices = knn_model.kneighbors(X)

    if not include_self:
        # sklearn returns the query itself as its first zero-distance neighbor.
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    rows = np.repeat(np.arange(n_points), indices.shape[1])
    adjacency = csr_matrix(
        (distances.reshape(-1), (rows, indices.reshape(-1))),
        shape=(n_points, n_points),
    )
    # Keep an undirected edge when either endpoint selected the other. For
    # distances, maximum is safe because a shared edge has the same value in
    # both directions and a missing sparse entry is represented by zero.
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.eliminate_zeros()
    return adjacency


def gaussian_kernel(adjacency, sigma=None, include_self=True):
    # Convert retained sparse distances to Gaussian affinity weights.

    adjacency = adjacency.tocsr()
    positive_distances = adjacency.data[adjacency.data > 0]
    if sigma is None:
        if positive_distances.size == 0:
            raise ValueError("Cannot infer sigma without positive distances.")
        sigma = float(np.mean(positive_distances))
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    kernel = adjacency.copy()
    kernel.data = np.exp(-(kernel.data**2) / (float(sigma) ** 2))
    kernel.eliminate_zeros()

    if include_self:
        kernel = kernel + eye(kernel.shape[0], format="csr")
    return kernel


def self_tuned_knn_affinity(
    features,
    n_neighbors,
    scale_neighbors=20,
    scale=8.0,
):
    # Build a union-symmetrized self-tuned kNN affinity from features.
    #
    # Each node's local scale is its ``scale_neighbors``-th neighbor distance.
    # An edge (i, j) receives exp(-d_ij^2 / (eta^2 sigma_i sigma_j)), where
    # ``scale`` is eta. This adapts the bandwidth to local sampling density.

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    n_points = features.shape[0]
    keep = min(int(n_neighbors), n_points - 1)
    scale_keep = min(max(int(scale_neighbors), 1), n_points - 1)
    if keep <= 0:
        raise ValueError("Need at least two points.")

    # One neighbor search supplies both graph edges and local bandwidths.
    search_keep = max(keep, scale_keep)
    model = NearestNeighbors(
        n_neighbors=search_keep + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    model.fit(features)
    distances, indices = model.kneighbors(features, return_distance=True)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    graph_neighbors = indices[:, :keep]
    graph_distances = distances[:, :keep]
    local_scales = distances[:, scale_keep - 1].copy()
    positive = local_scales[np.isfinite(local_scales) & (local_scales > 0)]
    if positive.size == 0:
        raise ValueError("No positive local bandwidths were found.")
    local_scales[
        ~np.isfinite(local_scales) | (local_scales <= 0)
    ] = float(np.median(positive))

    rows = np.repeat(np.arange(n_points), keep)
    columns = graph_neighbors.reshape(-1)
    edge_distances = graph_distances.reshape(-1)
    denominators = (
        float(scale) ** 2 * local_scales[rows] * local_scales[columns]
    )
    denominators = np.maximum(denominators, np.finfo(float).eps)
    weights = np.exp(-(edge_distances**2 / denominators))
    weights[~np.isfinite(weights)] = 0.0

    directed = csr_matrix(
        (weights, (rows, columns)),
        shape=(n_points, n_points),
    )
    # Union symmetrization retains an edge selected in either directed kNN
    # list. max also preserves the stronger weight if values differ slightly.
    affinity = directed.maximum(directed.T).tocsr()
    affinity.setdiag(0)
    affinity.eliminate_zeros()
    return affinity, graph_neighbors, local_scales


def self_tuned_distance_affinity(
    distances,
    n_neighbors,
    scale_neighbors=20,
    scale=8.0,
    chunk_size=512,
):
    # Build a self-tuned sparse affinity from a complete distance matrix.
    #
    # Rows are processed in chunks so neighbor selection does not require a
    # second full-size working copy of the distance matrix.

    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    n_points = distances.shape[0]
    keep = min(int(n_neighbors), n_points - 1)
    scale_keep = min(max(int(scale_neighbors), 1), n_points - 1)
    if keep <= 0:
        raise ValueError("Need at least two points.")

    neighbor_count = max(keep, scale_keep)
    graph_neighbors = np.empty((n_points, keep), dtype=int)
    graph_distances = np.empty((n_points, keep), dtype=float)
    local_scales = np.empty(n_points, dtype=float)

    for start in range(0, n_points, max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), n_points)
        block = distances[start:stop].copy()
        block[np.arange(stop - start), np.arange(start, stop)] = np.inf
        # argpartition finds the required neighborhood without sorting all n
        # distances. Only the selected entries are sorted afterward.
        neighbors = np.argpartition(
            block,
            neighbor_count - 1,
            axis=1,
        )[:, :neighbor_count]
        selected_distances = np.take_along_axis(block, neighbors, axis=1)
        order = np.argsort(selected_distances, axis=1)
        neighbors = np.take_along_axis(neighbors, order, axis=1)
        selected_distances = np.take_along_axis(
            selected_distances,
            order,
            axis=1,
        )

        graph_neighbors[start:stop] = neighbors[:, :keep]
        graph_distances[start:stop] = selected_distances[:, :keep]
        local_scales[start:stop] = selected_distances[:, scale_keep - 1]

    positive = local_scales[
        np.isfinite(local_scales) & (local_scales > 0)
    ]
    if positive.size == 0:
        raise ValueError("No positive local bandwidths were found.")
    local_scales[
        ~np.isfinite(local_scales) | (local_scales <= 0)
    ] = float(np.median(positive))

    rows = np.repeat(np.arange(n_points), keep)
    columns = graph_neighbors.reshape(-1)
    edge_distances = graph_distances.reshape(-1)
    denominators = (
        float(scale) ** 2 * local_scales[rows] * local_scales[columns]
    )
    denominators = np.maximum(denominators, np.finfo(float).eps)
    weights = np.exp(-(edge_distances**2 / denominators))
    weights[~np.isfinite(weights)] = 0.0

    directed = csr_matrix(
        (weights, (rows, columns)),
        shape=(n_points, n_points),
    )
    affinity = directed.maximum(directed.T).tocsr()
    affinity.setdiag(0)
    affinity.eliminate_zeros()
    return affinity, graph_neighbors, local_scales


def miller_knn_affinity(
    X,
    k=20,
    similarity="angular",
    gaussian_coefficient=4.0,
    release_knn_convention=True,
):
    # Build the fixed kNN Gaussian graph used by the PWLL release.
    #
    # The released PWLL experiments first find kNNs and then use directed
    # weights
    #
    #     exp(-gaussian_coefficient * d(i, j)^2 / d_k(i)^2).
    #
    # The directed matrix is arithmetically symmetrized. For HSI data, the
    # release performs neighbor search with angular distance, implemented here
    # as Euclidean distance between unit-normalized spectra.
    #
    # ``release_knn_convention=True`` reproduces the release's convention that
    # the requested neighbor count includes the point itself. Thus ``k=20``
    # retains at most 19 non-self directed neighbors.

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    n_points = X.shape[0]
    if n_points < 2:
        raise ValueError("Need at least two points.")
    if int(k) <= 1 and release_knn_convention:
        raise ValueError("k must be at least 2 when it includes the query point.")
    if int(k) <= 0:
        raise ValueError("k must be positive.")
    if gaussian_coefficient <= 0:
        raise ValueError("gaussian_coefficient must be positive.")
    if similarity not in {"angular", "euclidean"}:
        raise ValueError("similarity must be 'angular' or 'euclidean'.")

    features = X
    if similarity == "angular":
        # Euclidean distance between unit vectors is monotone in cosine angle,
        # reproducing angular neighbor ordering with sklearn's fast kNN.
        norms = np.linalg.norm(features, axis=1)
        features = np.zeros_like(features, dtype=float)
        positive = norms > 0
        features[positive] = X[positive] / norms[positive, np.newaxis]

    if release_knn_convention:
        search_count = min(int(k), n_points)
        keep = search_count - 1
    else:
        keep = min(int(k), n_points - 1)
        search_count = keep + 1
    if keep <= 0:
        raise ValueError("The selected k leaves no non-self neighbors.")

    model = NearestNeighbors(
        n_neighbors=search_count,
        metric="euclidean",
        n_jobs=-1,
    )
    model.fit(features)
    distances, indices = model.kneighbors(features, return_distance=True)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # The released kernel uses the final retained neighbor as node i's
    # directed bandwidth, rather than a global Gaussian sigma.
    local_scales = distances[:, -1].copy()
    positive_scales = local_scales[
        np.isfinite(local_scales) & (local_scales > 0)
    ]
    if positive_scales.size == 0:
        raise ValueError("No positive local kNN bandwidths were found.")
    fallback = float(np.median(positive_scales))
    local_scales[
        ~np.isfinite(local_scales) | (local_scales <= 0)
    ] = fallback

    weights = np.exp(
        -float(gaussian_coefficient)
        * distances**2
        / np.maximum(local_scales[:, np.newaxis] ** 2, np.finfo(float).eps)
    )
    weights[~np.isfinite(weights)] = 0.0

    rows = np.repeat(np.arange(n_points), keep)
    directed = csr_matrix(
        (weights.reshape(-1), (rows, indices.reshape(-1))),
        shape=(n_points, n_points),
    )
    # Match the reference release: arithmetic, not max, symmetrization.
    affinity = ((directed + directed.T) * 0.5).tocsr()
    affinity.setdiag(0)
    affinity.eliminate_zeros()
    return affinity, indices, local_scales


def fermat_adjacency(X, k, p):
    # Build the powered-edge graph used for discrete Fermat paths.

    if p <= 0:
        raise ValueError("p must be positive.")
    adjacency = knn_adjacency(X, k)
    adjacency.data = adjacency.data ** float(p)
    adjacency.eliminate_zeros()
    return adjacency


def fermat_distances(X, k, p, indices=None):
    # Compute sparse-graph Fermat shortest-path distances.
    #
    # Shortest paths are computed on powered edge costs ``||x_i-x_j||**p``.
    # The completed minimum path energy is transformed to ``D**(1/p)`` before
    # returning, following the discrete Fermat-distance definition.

    # scipy applies Dijkstra to powered edge costs and therefore returns the
    # minimum path energy sum ||x_i-x_j||^p.
    adjacency = fermat_adjacency(X, k, p)
    distances = shortest_path(
        adjacency,
        directed=False,
        indices=indices,
        unweighted=False,
    )
    
    distances = np.asarray(distances, dtype=float)
    positive = np.isfinite(distances) & (distances > 0)
    distances[positive] = distances[positive] ** (1.0 / float(p))
    return distances


def feature_knn_edge_cache(features, n_neighbors):
    # Cache edge endpoints and distances so bandwidth sweeps avoid kNN work.

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")

    n_points = features.shape[0]
    if n_points < 2:
        raise ValueError("Need at least two points.")

    neighbor_count = min(int(n_neighbors) + 1, n_points)
    knn_model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="euclidean",
        n_jobs=-1,
    )
    knn_model.fit(features)
    feature_distances, indices = knn_model.kneighbors(features)
    feature_distances = feature_distances[:, 1:]
    indices = indices[:, 1:]

    rows = np.repeat(np.arange(n_points), indices.shape[1])
    columns = indices.reshape(-1)
    selected_distances = feature_distances.reshape(-1)

    positive_distances = selected_distances[
        np.isfinite(selected_distances) & (selected_distances > 0)
    ]
    if positive_distances.size == 0:
        raise ValueError("No positive feature distances found.")

    return DistanceEdgeCache(
        n_points=n_points,
        rows=np.asarray(rows, dtype=int),
        columns=np.asarray(columns, dtype=int),
        distances=np.asarray(selected_distances, dtype=float),
        base_epsilon=float(np.mean(positive_distances)),
    )


def distance_edge_affinity_from_cache(edge_cache, epsilon_scale=1.0):
    # Reweight cached edges with a global Gaussian bandwidth.

    if epsilon_scale <= 0:
        raise ValueError("epsilon_scale must be positive.")

    epsilon = float(epsilon_scale) * float(edge_cache.base_epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    distances = np.asarray(edge_cache.distances, dtype=float)
    weights = np.exp(-((distances / epsilon) ** 2))
    weights[~np.isfinite(weights)] = 0.0
    directed = csr_matrix(
        (
            weights,
            (
                np.asarray(edge_cache.rows, dtype=int),
                np.asarray(edge_cache.columns, dtype=int),
            ),
        ),
        shape=(edge_cache.n_points, edge_cache.n_points),
    )
    affinity = directed.maximum(directed.T)
    affinity.setdiag(0)
    affinity.eliminate_zeros()
    return affinity, epsilon


def _replace_nonfinite_distances(distances):
    # Replace unreachable landmark distances with the largest finite value.

    distances = np.asarray(distances, dtype=float).copy()
    finite = np.isfinite(distances)
    if not np.any(finite):
        raise ValueError("No finite Fermat distances were found.")
    distances[~finite] = float(np.max(distances[finite]))
    return distances


def _landmark_mds_features(distances, landmarks, n_components):
    # extends classical MDS coordinates from landmark distances.
    #
    # ``distances`` has one row per landmark source and one column per data
    # point. Classical MDS embeds the landmark-to-landmark block; the
    # out-of-sample formula then projects every point from its distances to the
    # same landmarks.

    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    landmarks = np.asarray(landmarks, dtype=int)
    all_to_landmarks = _replace_nonfinite_distances(distances.T)
    landmark_distances = _replace_nonfinite_distances(distances[:, landmarks])
    landmark_distances = 0.5 * (landmark_distances + landmark_distances.T)

    # Double centering converts squared landmark distances into the Gram
    # matrix that would generate them in Euclidean coordinates.
    squared_landmarks = landmark_distances**2
    row_mean = squared_landmarks.mean(axis=1)
    grand_mean = float(squared_landmarks.mean())
    centered = -0.5 * (
        squared_landmarks
        - row_mean[:, np.newaxis]
        - row_mean[np.newaxis, :]
        + grand_mean
    )
    centered = 0.5 * (centered + centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    tolerance = np.finfo(float).eps * max(centered.shape) * max(
        float(np.max(np.abs(eigenvalues))),
        1.0,
    )
    # Negative eigenvalues measure non-Euclidean distortion and cannot define
    # real Euclidean coordinates, so classical MDS keeps positive directions.
    positive = np.flatnonzero(eigenvalues > tolerance)
    if positive.size == 0:
        raise ValueError("Landmark MDS produced no positive eigenvalues.")

    keep = positive[: min(int(n_components), positive.size)]
    values = eigenvalues[keep]
    vectors = eigenvectors[:, keep]

    # Apply the standard landmark out-of-sample extension to every point.
    squared_cross = all_to_landmarks**2
    cross_mean = squared_cross.mean(axis=1, keepdims=True)
    cross_gram = -0.5 * (
        squared_cross
        - cross_mean
        - row_mean[np.newaxis, :]
        + grand_mean
    )
    features = cross_gram @ (vectors / np.sqrt(values)[np.newaxis, :])
    features[landmarks] = vectors * np.sqrt(values)[np.newaxis, :]
    return features
