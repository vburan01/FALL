# Landmark and active-query selection used by the paper experiments.

import numpy as np
from sklearn.metrics import pairwise_distances


def farthest_point_indices(X, budget, start_index=None, random_state=0):
    # Select a geometrically diverse set by greedy farthest-point sampling.

    X = np.asarray(X)
    n_points = X.shape[0]
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if budget > n_points:
        raise ValueError("budget cannot exceed the number of points.")

    rng = np.random.default_rng(random_state)
    if start_index is None:
        start_index = int(rng.integers(n_points))
    if start_index < 0 or start_index >= n_points:
        raise ValueError("start_index is out of range.")

    selected = [start_index]
    # min_distances[i] is maintained as the distance from point i to its
    # nearest already selected landmark.
    min_distances = pairwise_distances(
        X,
        X[[start_index]],
        metric="euclidean",
    ).ravel()

    while len(selected) < budget:
        # Maximizing the nearest-landmark distance greedily covers unexplored
        # parts of feature space.
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        distances = pairwise_distances(
            X,
            X[[next_index]],
            metric="euclidean",
        ).ravel()
        min_distances = np.minimum(min_distances, distances)
        min_distances[selected] = 0.0

    return np.asarray(selected, dtype=int)


def landmark_indices(X, count, strategy="farthest", random_state=0):
    # Select the landmark nodes used by Landmark MDS.

    X = np.asarray(X)
    n_points = X.shape[0]
    if count <= 0:
        raise ValueError("count must be positive.")
    if count > n_points:
        raise ValueError("count cannot exceed the number of points.")
    if strategy not in {"farthest", "random"}:
        raise ValueError("strategy must be 'farthest' or 'random'.")

    rng = np.random.default_rng(random_state)
    if strategy == "random":
        return rng.choice(n_points, size=count, replace=False)
    return farthest_point_indices(
        X,
        budget=count,
        start_index=int(rng.integers(n_points)),
        random_state=random_state,
    )


def minimum_norm_acquisition(scores, candidate_indices=None):
    # Score candidates by PWLL minimum-norm uncertainty.
    #
    # A confident class score is close to a one-hot vector and has large L2
    # norm. Returning ``1 - norm`` makes uncertain, low-norm nodes maximize the
    # acquisition function.

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional array.")

    if candidate_indices is None:
        candidate_indices = np.arange(scores.shape[0])
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=int)

    if np.any(candidate_indices < 0) or np.any(
        candidate_indices >= scores.shape[0]
    ):
        raise ValueError("candidate_indices contains an out-of-range index.")

    return 1.0 - np.linalg.norm(scores[candidate_indices], axis=1)
