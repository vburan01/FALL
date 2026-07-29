# Harmonic and Poisson-reweighted graph label propagation.
#
# PWLL first solves a compatible graph Poisson equation to amplify weights near
# the labeled set, then performs tau-regularized harmonic extension with the
# observed one-hot labels as Dirichlet boundary values.

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import cg, spsolve


def _prepare_labels(affinity, labeled_indices, labeled_labels, classes):
    # Validate labels and remove any accidental background observations.

    n_points = affinity.shape[0]
    labeled_indices = np.asarray(labeled_indices, dtype=int)
    labeled_labels = np.asarray(labeled_labels)

    if labeled_indices.size != labeled_labels.size:
        raise ValueError("labeled_indices and labeled_labels must have the same length.")
    if labeled_indices.size == 0:
        raise ValueError("Need at least one labeled point.")
    if np.any(labeled_indices < 0) or np.any(labeled_indices >= n_points):
        raise ValueError("labeled_indices contains an out-of-range index.")

    keep = labeled_labels != 0
    labeled_indices = labeled_indices[keep]
    labeled_labels = labeled_labels[keep]
    if labeled_indices.size == 0:
        raise ValueError("Need at least one non-background labeled point.")

    if classes is None:
        classes = np.unique(labeled_labels)
    else:
        classes = np.asarray(classes)

    if classes.size == 0:
        raise ValueError("classes must contain at least one class.")

    return labeled_indices, labeled_labels, classes


def _tau_diagonal(tau, n_points, unlabeled_indices):
    # Construct tau I, or its node-wise diagonal analogue, on U.

    tau_values = np.asarray(tau, dtype=float)
    if tau_values.ndim == 0:
        if tau_values < 0:
            raise ValueError("tau must be nonnegative.")
        return float(tau_values) * eye(unlabeled_indices.size, format="csr")

    if tau_values.ndim != 1 or tau_values.size != n_points:
        raise ValueError("tau must be a scalar or one value per graph node.")
    if np.any(tau_values < 0):
        raise ValueError("tau must be nonnegative.")
    return diags(tau_values[unlabeled_indices], format="csr")


def pwll_tau_decay(
    initial_tau,
    query_iteration,
    cluster_count,
    epsilon=1e-9,
):
    # Evaluate the geometric PWLL tau-decay schedule.
    #
    # Tau falls from ``initial_tau`` to ``epsilon`` over ``2*C`` acquisitions
    # and is set exactly to zero afterward.

    initial_tau = float(initial_tau)
    query_iteration = int(query_iteration)
    cluster_count = int(cluster_count)
    epsilon = float(epsilon)

    if initial_tau < 0:
        raise ValueError("initial_tau must be nonnegative.")
    if query_iteration < 0:
        raise ValueError("query_iteration must be nonnegative.")
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive.")
    if initial_tau == 0:
        return 0.0
    if epsilon <= 0 or epsilon >= initial_tau:
        raise ValueError("epsilon must satisfy 0 < epsilon < initial_tau.")

    decay_steps = 2 * cluster_count
    if query_iteration >= decay_steps:
        return 0.0

    decay_rate = (epsilon / initial_tau) ** (1.0 / decay_steps)
    return initial_tau * decay_rate**query_iteration


def harmonic_label_propagation(
    affinity,
    labeled_indices,
    labeled_labels,
    classes=None,
    tau=0.0,
):
    # Solve the block harmonic system for unlabeled class scores.
    #
    # With labeled scores F_L fixed, the first-order condition is
    # (L_UU + tau I) F_U = -L_UL F_L. The largest component of each resulting
    # score vector determines the predicted class.

    affinity = affinity.tocsr()
    n_points = affinity.shape[0]
    labeled_indices, labeled_labels, classes = _prepare_labels(
        affinity,
        labeled_indices,
        labeled_labels,
        classes,
    )

    labeled_mask = np.zeros(n_points, dtype=bool)
    labeled_mask[labeled_indices] = True
    unlabeled_indices = np.flatnonzero(~labeled_mask)

    # Boundary values are one-hot rows in the global class order.
    scores = np.zeros((n_points, classes.size), dtype=float)
    for class_index, class_label in enumerate(classes):
        scores[labeled_indices, class_index] = labeled_labels == class_label

    if unlabeled_indices.size == 0:
        predictions = classes[np.argmax(scores, axis=1)]
        return predictions, scores

    # The submitted experiments use the combinatorial Laplacian D - W.
    L = laplacian(affinity, normed=False).tocsr()
    L_uu = L[unlabeled_indices][:, unlabeled_indices]
    L_ul = L[unlabeled_indices][:, labeled_indices]
    labeled_scores = scores[labeled_indices]

    tau_diagonal = _tau_diagonal(tau, n_points, unlabeled_indices)
    system = L_uu + tau_diagonal
    rhs = -L_ul @ labeled_scores
    # All class right-hand sides share the same sparse system matrix.
    unlabeled_scores = spsolve(system, rhs)
    if unlabeled_scores.ndim == 1:
        unlabeled_scores = unlabeled_scores[:, np.newaxis]
    scores[unlabeled_indices] = unlabeled_scores

    predictions = classes[np.argmax(scores, axis=1)]
    predictions[labeled_indices] = labeled_labels

    return predictions, scores


def poisson_reweighted_affinity(
    affinity,
    labeled_indices,
    tolerance=1e-5,
    max_iterations=None,
    gamma_floor=1e-5,
    poisson_gauge="grounded",
):
    # Construct the Poisson-reweighted affinity ``Gamma W Gamma``.
    #
    # The source has +1 at labeled nodes and is centered to sum to zero, making
    # it compatible with the graph Laplacian. A gauge removes the additive
    # constant ambiguity before a uniform shift makes gamma strictly positive.

    affinity = affinity.tocsr()
    n_points = affinity.shape[0]
    if affinity.shape[1] != n_points:
        raise ValueError("affinity must be a square matrix.")
    if gamma_floor <= 0:
        raise ValueError("gamma_floor must be positive.")
    if poisson_gauge not in {"grounded", "original"}:
        raise ValueError("poisson_gauge must be 'grounded' or 'original'.")

    labeled_indices = np.unique(np.asarray(labeled_indices, dtype=int))
    if labeled_indices.size == 0:
        raise ValueError("Need at least one labeled point.")
    if np.any(labeled_indices < 0) or np.any(labeled_indices >= n_points):
        raise ValueError("labeled_indices contains an out-of-range index.")

    n_components = connected_components(
        affinity,
        directed=False,
        return_labels=False,
    )
    if n_components != 1:
        raise ValueError("Poisson reweighting requires a connected affinity graph.")

    # Centering is required because every Laplacian row sums to zero, so its
    # range is orthogonal to the constant vector on a connected graph.
    source = np.zeros(n_points, dtype=float)
    source[labeled_indices] = 1.0
    source -= source.mean()

    L = laplacian(affinity, normed=False).tocsr()

    if poisson_gauge == "grounded":
        # Fix one node to zero to remove the Laplacian's additive-constant
        # nullspace. The omitted equation follows from sum(source) = 0 and the
        # zero row sums of L.
        grounded_system = L[:-1, :-1]
        grounded_source = source[:-1]
        diagonal = grounded_system.diagonal()
        # Jacobi preconditioning is inexpensive and improves CG convergence
        # when Poisson-reweighted graphs have heterogeneous degrees.
        preconditioner = diags(1.0 / diagonal, format="csr")
        if max_iterations is None:
            max_iterations = max(1000, 10 * grounded_system.shape[0])

        grounded_gamma, info = cg(
            grounded_system,
            grounded_source,
            rtol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
            M=preconditioner,
        )
        gamma = np.zeros(n_points, dtype=float)
        gamma[:-1] = grounded_gamma
    else:
        # Match the original GraphLearning/PWLL style more closely: solve the
        # compatible singular Poisson system directly, then remove any numerical
        # constant drift before applying the positivity shift below.
        if max_iterations is None:
            max_iterations = max(1000, 10 * n_points)
        gamma, info = cg(
            L,
            source,
            rtol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
        )
        gamma = gamma - gamma.mean()

    if info > 0:
        raise RuntimeError(
            "Poisson reweighting did not converge within "
            f"{info} conjugate-gradient iterations."
        )
    if info < 0:
        raise RuntimeError("Poisson reweighting conjugate gradient failed.")
    if not np.all(np.isfinite(gamma)):
        raise RuntimeError("Poisson reweighting produced non-finite values.")

    # Poisson solutions are defined up to a constant. This shift preserves all
    # pairwise potential differences while ensuring positive edge multipliers.
    gamma = gamma - gamma.min() + gamma_floor
    reweighted = affinity.multiply(gamma[:, np.newaxis])
    reweighted = reweighted.multiply(gamma[np.newaxis, :]).tocsr()
    reweighted.eliminate_zeros()

    return reweighted, gamma


def poisson_reweighted_laplace_learning(
    affinity,
    labeled_indices,
    labeled_labels,
    classes=None,
    reweight_tolerance=1e-5,
    reweight_max_iterations=None,
    gamma_floor=1e-5,
    return_reweighting=False,
    tau=0.0,
    poisson_gauge="grounded",
):
    # Run the complete PWLL-tau pipeline on a fixed base affinity.

    affinity = affinity.tocsr()
    labeled_indices, labeled_labels, classes = _prepare_labels(
        affinity,
        labeled_indices,
        labeled_labels,
        classes,
    )
    # Reweighting depends on the currently labeled set and is recomputed at
    # each active-learning round.
    reweighted, gamma = poisson_reweighted_affinity(
        affinity,
        labeled_indices,
        tolerance=reweight_tolerance,
        max_iterations=reweight_max_iterations,
        gamma_floor=gamma_floor,
        poisson_gauge=poisson_gauge,
    )
    predictions, scores = harmonic_label_propagation(
        reweighted,
        labeled_indices=labeled_indices,
        labeled_labels=labeled_labels,
        classes=classes,
        tau=tau,
    )

    if return_reweighting:
        return predictions, scores, reweighted, gamma
    return predictions, scores
