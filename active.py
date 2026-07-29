# Exact and approximate leave-one-out selection of the Fermat exponent.
#
# ELOO removes each eligible label and reruns PWLL. ALOO freezes the
# full-labeled-set Poisson reweighting and uses a Kron-reduced operator to
# estimate every held-out prediction from one graph solve per candidate p.

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import spsolve

from .label_propagation import (
    poisson_reweighted_affinity,
    poisson_reweighted_laplace_learning,
)


def _clip_normalize_rows(scores):
    # Convert unconstrained harmonic scores into probability-like rows.
    #
    # Harmonic scores can be negative and need not sum to one. The paper's
    # scoring rule clips negative entries, normalizes positive mass, and uses a
    # uniform row when all entries are numerically zero.

    scores = np.asarray(scores, dtype=float)
    normalized = np.clip(scores, 0.0, None)
    if normalized.ndim == 1:
        total = normalized.sum()
        if total <= np.finfo(float).eps:
            return np.full(scores.shape, 1.0 / scores.size)
        return normalized / total

    row_sums = normalized.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > np.finfo(float).eps
    result = np.full_like(normalized, 1.0 / scores.shape[1], dtype=float)
    result[valid] = normalized[valid] / row_sums[valid]
    return result


def _one_hot_labels(labels, classes):
    # Encode class identifiers using the fixed global class order.

    class_to_position = {label: index for index, label in enumerate(classes)}
    values = np.zeros((labels.size, classes.size), dtype=float)
    for row, label in enumerate(labels):
        values[row, class_to_position[label]] = 1.0
    return values, class_to_position


def _aggregate_loo_class_scores(class_losses, class_margins):
    # Average all eligible held-out losses and margins uniformly.

    nonempty_losses = [losses for losses in class_losses.values() if len(losses)]
    nonempty_margins = [margins for margins in class_margins.values() if len(margins)]
    if not nonempty_losses:
        return np.nan, np.nan

    loss = float(
        np.mean(np.concatenate([np.asarray(losses) for losses in nonempty_losses]))
    )
    margin = float(
        np.mean(np.concatenate([np.asarray(margins) for margins in nonempty_margins]))
    )
    return loss, margin


def _eligible_validation_positions(labeled_labels, min_validation_classes):
    # Find labels that can be held out without removing their class entirely.

    observed_classes, counts = np.unique(labeled_labels, return_counts=True)
    # A class needs two observations: one held-out point and at least one
    # training point that still identifies the class.
    eligible_classes = observed_classes[counts >= 2]
    if eligible_classes.size < min_validation_classes:
        return eligible_classes, np.array([], dtype=int)

    eligible_mask = np.isin(labeled_labels, eligible_classes)
    return eligible_classes, np.flatnonzero(eligible_mask)


def _prepare_labeled_data(labeled_indices, labeled_labels, classes):
    # Discard background labels and standardize LOO inputs as arrays.

    labeled_indices = np.asarray(labeled_indices, dtype=int)
    labeled_labels = np.asarray(labeled_labels)
    classes = np.asarray(classes)

    keep = labeled_labels != 0
    return labeled_indices[keep], labeled_labels[keep], classes


def _validate_loo_options(
    candidate_affinities,
    min_validation_classes,
    margin_weight,
):
    # Validate options shared by exact and approximate selectors.

    if not candidate_affinities:
        raise ValueError("candidate_affinities cannot be empty.")
    if min_validation_classes <= 0:
        raise ValueError("min_validation_classes must be positive.")
    if margin_weight < 0:
        raise ValueError("margin_weight must be nonnegative.")


def _best_p_from_scores(scores, current_p):
    # Choose the best score with deterministic, conservative tie-breaking.

    if not scores:
        return float(current_p)
    return float(
        max(
            sorted(scores),
            key=lambda p: (
                scores[p],
                # Prefer the smallest change from the current geometry, then
                # the smaller exponent, when validation scores are tied.
                -abs(p - float(current_p)),
                -p,
            ),
        )
    )


def select_graph_by_leave_one_out(
    candidate_affinities,
    current_p,
    labeled_indices,
    labeled_labels,
    classes,
    min_validation_classes=2,
    reweight_tolerance=1e-5,
    reweight_max_iterations=None,
    gamma_floor=1e-5,
    poisson_gauge="grounded",
    tau=0.0,
    margin_weight=0.0,
):
    # Select p by exact LOO PWLL loss and optional margin reward.
    #
    # Each eligible labeled point is removed in turn. PWLL, including Poisson
    # reweighting, is recomputed using the remaining labels, making this the
    # reference implementation against which ALOO is compared.

    _validate_loo_options(
        candidate_affinities,
        min_validation_classes,
        margin_weight,
    )

    labeled_indices, labeled_labels, classes = _prepare_labeled_data(
        labeled_indices,
        labeled_labels,
        classes,
    )
    if labeled_indices.size < 2:
        return float(current_p), {}, 0

    eligible_classes, validation_positions = _eligible_validation_positions(
        labeled_labels,
        min_validation_classes,
    )
    if validation_positions.size == 0:
        return float(current_p), {}, 0

    class_to_position = {label: index for index, label in enumerate(classes)}
    scores = {}
    for p, affinity in candidate_affinities.items():
        class_losses = {label: [] for label in eligible_classes}
        class_margins = {label: [] for label in eligible_classes}

        for validation_position in validation_positions:
            validation_index = labeled_indices[validation_position]
            validation_label = labeled_labels[validation_position]
            training_mask = np.ones(labeled_indices.size, dtype=bool)
            training_mask[validation_position] = False

            # Exact LOO changes both the boundary labels and the
            # label-dependent Poisson reweighting for every held-out point.
            _, model_scores = poisson_reweighted_laplace_learning(
                affinity,
                labeled_indices=labeled_indices[training_mask],
                labeled_labels=labeled_labels[training_mask],
                classes=classes,
                reweight_tolerance=reweight_tolerance,
                reweight_max_iterations=reweight_max_iterations,
                gamma_floor=gamma_floor,
                poisson_gauge=poisson_gauge,
                tau=tau,
            )

            row = np.asarray(model_scores[validation_index], dtype=float)
            row = _clip_normalize_rows(row)

            true_position = class_to_position[validation_label]
            target = np.zeros(classes.size, dtype=float)
            target[true_position] = 1.0
            loss = float(np.sum((row - target) ** 2))

            # Positive margin means the true class exceeds its strongest
            # competitor. It acts as a confidence reward beside squared loss.
            competitors = row.copy()
            competitors[true_position] = -np.inf
            margin = float(row[true_position] - np.max(competitors))

            class_losses[validation_label].append(loss)
            class_margins[validation_label].append(margin)

        loss, margin = _aggregate_loo_class_scores(class_losses, class_margins)
        scores[float(p)] = -loss + margin_weight * margin

    return _best_p_from_scores(scores, current_p), scores, int(validation_positions.size)


def _tau_values_for_indices(tau, n_points, indices):
    # Restrict scalar or node-wise tau values to a graph subset.

    if np.isscalar(tau):
        return np.full(indices.size, float(tau), dtype=float)

    tau = np.asarray(tau, dtype=float)
    if tau.shape != (n_points,):
        raise ValueError("vector tau must have one entry per graph node.")
    return tau[indices]


def _pwll_operator_for_aloocv(
    affinity,
    labeled_indices,
    reweight_tolerance,
    reweight_max_iterations,
    gamma_floor,
    poisson_gauge,
):
    # Build the frozen Poisson-reweighted Laplacian used by ALOO.

    working_affinity, _ = poisson_reweighted_affinity(
        affinity,
        labeled_indices=labeled_indices,
        tolerance=reweight_tolerance,
        max_iterations=reweight_max_iterations,
        gamma_floor=gamma_floor,
        poisson_gauge=poisson_gauge,
    )
    return laplacian(working_affinity, normed=False).tocsr()


def approximate_leave_one_out_scores(
    affinity,
    labeled_indices,
    labeled_labels,
    classes,
    min_validation_classes=2,
    reweight_tolerance=1e-5,
    reweight_max_iterations=None,
    gamma_floor=1e-5,
    poisson_gauge="grounded",
    tau=0.0,
    margin_weight=0.0,
):
    # Approximate PWLL leave-one-out scores by Kron reduction.
    #
    # The Poisson-reweighted graph is computed once from the complete labeled
    # set. Unlabeled nodes are then eliminated with a Schur complement, yielding
    # an effective operator on labeled nodes. Releasing each label is a cheap
    # row-wise solve on that reduced operator.

    _validate_loo_options(
        {0: affinity},
        min_validation_classes,
        margin_weight,
    )

    affinity = affinity.tocsr()
    n_points = affinity.shape[0]
    labeled_indices, labeled_labels, classes = _prepare_labeled_data(
        labeled_indices,
        labeled_labels,
        classes,
    )
    if labeled_indices.size < 2:
        return {}, 0

    eligible_classes, validation_positions = _eligible_validation_positions(
        labeled_labels,
        min_validation_classes,
    )
    if validation_positions.size == 0:
        return {}, 0

    operator = _pwll_operator_for_aloocv(
        affinity,
        labeled_indices=labeled_indices,
        reweight_tolerance=reweight_tolerance,
        reweight_max_iterations=reweight_max_iterations,
        gamma_floor=gamma_floor,
        poisson_gauge=poisson_gauge,
    )

    labeled_mask = np.zeros(n_points, dtype=bool)
    labeled_mask[labeled_indices] = True
    unlabeled_indices = np.flatnonzero(~labeled_mask)

    # Partition the frozen operator into labeled and unlabeled blocks.
    L_ll = operator[labeled_indices][:, labeled_indices].toarray()
    if unlabeled_indices.size:
        L_uu = operator[unlabeled_indices][:, unlabeled_indices].tocsr()
        L_ul = operator[unlabeled_indices][:, labeled_indices]
        unlabeled_tau = _tau_values_for_indices(tau, n_points, unlabeled_indices)
        system = L_uu + diags(unlabeled_tau, format="csr")
        solved = spsolve(system, L_ul.toarray())
        if solved.ndim == 1:
            solved = solved[:, np.newaxis]
        # S = L_LL - L_LU (L_UU + tau I)^(-1) L_UL is the
        # Kron-reduced operator that preserves minimized graph energy on L.
        schur = L_ll - (L_ul.T @ solved)
    else:
        schur = L_ll

    labeled_tau = _tau_values_for_indices(tau, n_points, labeled_indices)
    diagonal = np.diag(schur) + labeled_tau
    eps = np.finfo(float).eps
    diagonal = np.where(np.abs(diagonal) > eps, diagonal, eps)

    label_values, class_to_position = _one_hot_labels(labeled_labels, classes)
    schur_times_labels = schur @ label_values
    # Remove each point's own fixed one-hot contribution. The remaining row
    # is exactly the contribution from L \ {a} in the reduced first-order
    # condition S_aa F_a + S_aR Y_R = 0.
    numerators = schur_times_labels - np.diag(schur)[:, np.newaxis] * label_values
    loo_scores = -numerators / diagonal[:, np.newaxis]
    loo_scores = _clip_normalize_rows(loo_scores)

    class_losses = {label: [] for label in eligible_classes}
    class_margins = {label: [] for label in eligible_classes}

    for position in validation_positions:
        validation_label = labeled_labels[position]
        true_position = class_to_position[validation_label]
        row = loo_scores[position]

        target = np.zeros(classes.size, dtype=float)
        target[true_position] = 1.0
        loss = float(np.sum((row - target) ** 2))

        competitors = row.copy()
        competitors[true_position] = -np.inf
        margin = float(row[true_position] - np.max(competitors))

        class_losses[validation_label].append(loss)
        class_margins[validation_label].append(margin)

    loss, margin = _aggregate_loo_class_scores(class_losses, class_margins)
    score = -loss + margin_weight * margin
    return {"score": score, "loss": loss, "margin": margin}, int(
        validation_positions.size
    )


def select_graph_by_approximate_leave_one_out(
    candidate_affinities,
    current_p,
    labeled_indices,
    labeled_labels,
    classes,
    min_validation_classes=2,
    reweight_tolerance=1e-5,
    reweight_max_iterations=None,
    gamma_floor=1e-5,
    poisson_gauge="grounded",
    tau=0.0,
    margin_weight=0.0,
):
    # Evaluate ALOO for every candidate graph and select its best exponent.

    if not candidate_affinities:
        raise ValueError("candidate_affinities cannot be empty.")

    scores = {}
    validation_count = 0
    for p, affinity in candidate_affinities.items():
        result, count = approximate_leave_one_out_scores(
            affinity,
            labeled_indices=labeled_indices,
            labeled_labels=labeled_labels,
            classes=classes,
            min_validation_classes=min_validation_classes,
            reweight_tolerance=reweight_tolerance,
            reweight_max_iterations=reweight_max_iterations,
            gamma_floor=gamma_floor,
            poisson_gauge=poisson_gauge,
            tau=tau,
            margin_weight=margin_weight,
        )
        if result:
            scores[float(p)] = float(result["score"])
            validation_count = max(validation_count, count)

    return _best_p_from_scores(scores, current_p), scores, int(validation_count)
