# Reproducible PWLL-tau, FALL, and A-FALL experiment orchestration.
#
# This module owns the full paper protocol: graph construction, sequential
# minimum-norm querying, optional ALOO/ELOO updates of the Fermat exponent,
# foreground-only OA/AA evaluation, timing, summaries, and saved artifacts.
# Graph construction is cached across seeds because it is deterministic for a
# fixed configuration.

from dataclasses import asdict, dataclass
from itertools import product
import json
import platform
from pathlib import Path
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import sklearn

from .active import (
    select_graph_by_approximate_leave_one_out,
    select_graph_by_leave_one_out,
)
from .graph import (
    _landmark_mds_features,
    fermat_adjacency,
    fermat_distances,
    miller_knn_affinity,
    self_tuned_distance_affinity,
    self_tuned_knn_affinity,
)
from .label_propagation import (
    poisson_reweighted_laplace_learning,
    pwll_tau_decay,
)
from .preprocessing import load_hsi_dataset
from .query import landmark_indices, minimum_norm_acquisition


# Dataset presets expose the paper's default budget and exponent choices while
# leaving all sweepable parameters available through ``run_FALL.py``.
@dataclass(frozen=True)
class DatasetPreset:
    # Dataset-specific settings used to produce the submitted table.

    budget: int
    fall_p: float
    afall_initial_p: float
    afall_candidate_p: tuple


PAPER_PRESETS = {
    "salinasA": DatasetPreset(
        budget=20,
        fall_p=8.0,
        afall_initial_p=10.0,
        afall_candidate_p=(1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0),
    ),
    "paviaU_crop": DatasetPreset(
        budget=30,
        fall_p=8.0,
        afall_initial_p=4.0,
        afall_candidate_p=(
            1.25,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
        ),
    ),
}


HARD_CODED_SETTINGS = {
    "normalization": "spectral_band",
    "initialization": "one random foreground label",
    "query_pool": "foreground pixels only",
    "graph_scope": "all pixels in the selected scene",
    "acquisition": "minimum norm",
    "laplacian": "combinatorial (no degree normalization)",
    "self_tune_neighbors_k_sigma": 20,
    "landmark_strategy": "farthest",
    "landmark_seed": 0,
    "tau_endpoint": 1e-9,
    "gamma_floor": 1e-5,
    "aloo_min_validation_classes": 2,
    "loo_margin_weight_default": 0.02,
    "exact_dense_limit": 16000,
    "exact_chunk_size": 512,
}


# These columns uniquely describe a configuration and are copied into every
# raw row and final summary for auditability.
CONFIG_COLUMNS = [
    "algorithm",
    "dataset",
    "setting_id",
    "max_budget",
    "initial_tau",
    "pwll_k",
    "gaussian_coefficient",
    "p",
    "initial_p",
    "candidate_p_values",
    "fermat_k",
    "graph_neighbors",
    "kernel_scale",
    "landmark_count",
    "mds_dimension",
    "update_period",
    "loo_margin_weight",
    "p_selection_method",
]


def accuracy(predictions, ground_truth):
    # Compute overall accuracy on non-background ground-truth labels.

    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    if predictions.shape != ground_truth.shape:
        raise ValueError("predictions and ground_truth must have the same shape.")

    foreground = ground_truth != 0
    if not np.any(foreground):
        raise ValueError("ground_truth has no non-background labels.")
    return float(np.mean(predictions[foreground] == ground_truth[foreground]))


def average_accuracy(predictions, ground_truth, classes):
    # Compute the mean of the foreground per-class accuracies.

    return float(
        np.mean(
            [
                np.mean(predictions[ground_truth == label] == label)
                for label in classes
            ]
        )
    )


def _float_tag(value):
    # Encode a float compactly for deterministic setting IDs.

    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _safe_name(text):
    # Restrict generated setting IDs to filename-safe characters.

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return text.strip("_")


def _setting_id(setting):
    # Build a stable identifier from every graph- or solver-changing value.

    algorithm = setting["algorithm"]
    if algorithm == "pwll-tau":
        parts = [
            algorithm,
            f"k{setting['pwll_k']}",
            f"coef{_float_tag(setting['gaussian_coefficient'])}",
            f"tau{_float_tag(setting['initial_tau'])}",
            f"B{setting['max_budget']}",
        ]
    elif algorithm == "fall":
        parts = [
            algorithm,
            f"p{_float_tag(setting['p'])}",
            f"kF{setting['fermat_k']}",
            f"kG{setting['graph_neighbors']}",
            f"eta{_float_tag(setting['kernel_scale'])}",
            f"tau{_float_tag(setting['initial_tau'])}",
            f"B{setting['max_budget']}",
        ]
    else:
        parts = [
            algorithm,
            f"p0{_float_tag(setting['initial_p'])}",
            f"kF{setting['fermat_k']}",
            f"kG{setting['graph_neighbors']}",
            f"eta{_float_tag(setting['kernel_scale'])}",
            f"m{setting['landmark_count']}",
            f"r{setting['mds_dimension']}",
            f"T{setting['update_period']}",
            f"lam{_float_tag(setting['loo_margin_weight'])}",
            setting["p_selection_method"],
            f"tau{_float_tag(setting['initial_tau'])}",
            f"B{setting['max_budget']}",
        ]
    return _safe_name("__".join(parts))


def build_settings(args):
    # Expand scalar/list CLI values into the requested Cartesian grid.

    settings = []
    common = product(args.budget_values, args.initial_tau_values)
    common = list(common)

    if "pwll-tau" in args.algorithms:
        for (budget, tau), k, coefficient in product(
            common,
            args.pwll_k_values,
            args.pwll_gaussian_coefficient_values,
        ):
            setting = {
                "algorithm": "pwll-tau",
                "dataset": args.dataset,
                "max_budget": int(budget),
                "initial_tau": float(tau),
                "pwll_k": int(k),
                "gaussian_coefficient": float(coefficient),
                "p": None,
                "initial_p": None,
                "candidate_p_values": "",
                "fermat_k": None,
                "graph_neighbors": None,
                "kernel_scale": None,
                "landmark_count": None,
                "mds_dimension": None,
                "update_period": None,
                "loo_margin_weight": None,
                "p_selection_method": None,
            }
            setting["setting_id"] = _setting_id(setting)
            settings.append(setting)

    if "fall" in args.algorithms:
        for (
            (budget, tau),
            p,
            fermat_k,
            graph_neighbors,
            kernel_scale,
        ) in product(
            common,
            args.fall_p_values,
            args.fermat_k_values,
            args.graph_neighbor_values,
            args.kernel_scale_values,
        ):
            setting = {
                "algorithm": "fall",
                "dataset": args.dataset,
                "max_budget": int(budget),
                "initial_tau": float(tau),
                "pwll_k": None,
                "gaussian_coefficient": None,
                "p": float(p),
                "initial_p": None,
                "candidate_p_values": "",
                "fermat_k": int(fermat_k),
                "graph_neighbors": int(graph_neighbors),
                "kernel_scale": float(kernel_scale),
                "landmark_count": None,
                "mds_dimension": None,
                "update_period": None,
                "loo_margin_weight": None,
                "p_selection_method": None,
            }
            setting["setting_id"] = _setting_id(setting)
            settings.append(setting)

    if "a-fall" in args.algorithms:
        candidate_text = ";".join(
            f"{value:g}" for value in args.candidate_p_values
        )
        for (
            (budget, tau),
            initial_p,
            fermat_k,
            graph_neighbors,
            kernel_scale,
            landmark_count,
            mds_dimension,
            update_period,
            loo_margin_weight,
            p_selection_method,
        ) in product(
            common,
            args.afall_initial_p_values,
            args.fermat_k_values,
            args.graph_neighbor_values,
            args.kernel_scale_values,
            args.landmark_count_values,
            args.mds_dimension_values,
            args.update_period_values,
            args.loo_margin_weight_values,
            args.p_selection_methods,
        ):
            setting = {
                "algorithm": "a-fall",
                "dataset": args.dataset,
                "max_budget": int(budget),
                "initial_tau": float(tau),
                "pwll_k": None,
                "gaussian_coefficient": None,
                "p": None,
                "initial_p": float(initial_p),
                "candidate_p_values": candidate_text,
                "candidate_p_tuple": tuple(args.candidate_p_values),
                "fermat_k": int(fermat_k),
                "graph_neighbors": int(graph_neighbors),
                "kernel_scale": float(kernel_scale),
                "landmark_count": int(landmark_count),
                "mds_dimension": int(mds_dimension),
                "update_period": int(update_period),
                "loo_margin_weight": float(loo_margin_weight),
                "p_selection_method": p_selection_method,
            }
            setting["setting_id"] = _setting_id(setting)
            settings.append(setting)

    return settings


def _require_connected(affinity, description):
    # Reject disconnected graphs instead of silently adding bridge edges.

    components = connected_components(
        affinity,
        directed=False,
        return_labels=False,
    )
    if components != 1:
        raise ValueError(
            f"{description} produced {components} connected components."
        )


def _solver_options(dataset, algorithm):
    # Return numerical settings used in the submitted experiment protocol.

    if algorithm == "pwll-tau":
        return {
            "reweight_tolerance": 1e-5,
            "reweight_max_iterations": None,
            "poisson_gauge": "original",
        }
    if algorithm == "fall":
        return {
            "reweight_tolerance": (
                1e-5 if dataset == "salinasA" else 1e-4
            ),
            "reweight_max_iterations": 250000,
            "poisson_gauge": "grounded",
        }
    return {
        "reweight_tolerance": 1e-5 if dataset == "salinasA" else 1e-4,
        "reweight_max_iterations": (
            None if dataset == "salinasA" else 250000
        ),
        "poisson_gauge": "grounded",
    }


def _build_pwll_graph(X, setting):
    # Build the released PWLL angular-kNN affinity and time the full step.

    start = time.perf_counter()
    affinity, _, local_scales = miller_knn_affinity(
        X,
        k=setting["pwll_k"],
        similarity="angular",
        gaussian_coefficient=setting["gaussian_coefficient"],
        release_knn_convention=True,
    )
    seconds = time.perf_counter() - start
    _require_connected(affinity, "PWLL-tau graph")
    return {
        "affinity": affinity,
        "graph_seconds": seconds,
        "local_scale_median": float(np.median(local_scales)),
    }


def _chunked_exact_affinity(X, setting):
    # Build an exact-Fermat propagation graph without storing dense NxN paths.
    #
    # Dijkstra distances are computed in source chunks. Each row retains only
    # the neighbors needed by the final propagation graph and its self-tuned
    # local bandwidth.

    n_points = X.shape[0]
    keep = min(setting["graph_neighbors"], n_points - 1)
    scale_keep = min(
        HARD_CODED_SETTINGS["self_tune_neighbors_k_sigma"],
        n_points - 1,
    )
    neighbor_count = max(keep, scale_keep)
    chunk_size = HARD_CODED_SETTINGS["exact_chunk_size"]

    adjacency = fermat_adjacency(
        X,
        k=setting["fermat_k"],
        p=setting["p"],
    )
    graph_neighbors = np.empty((n_points, keep), dtype=int)
    graph_distances = np.empty((n_points, keep), dtype=float)
    local_scales = np.empty(n_points, dtype=float)

    indices = np.arange(n_points)
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        sources = indices[start:stop]
        block = np.asarray(
            shortest_path(
                adjacency,
                directed=False,
                indices=sources,
                unweighted=False,
            ),
            dtype=float,
        )
        # Dijkstra minimizes the sum of powered edge lengths. The outer p-th
        # root is applied only after shortest-path minimization.
        positive = np.isfinite(block) & (block > 0)
        block[positive] = block[positive] ** (1.0 / setting["p"])
        block[np.arange(stop - start), sources] = np.inf

        # Partial sorting avoids sorting all n distances for every source.
        neighbors = np.argpartition(
            block,
            neighbor_count - 1,
            axis=1,
        )[:, :neighbor_count]
        distances = np.take_along_axis(block, neighbors, axis=1)
        order = np.argsort(distances, axis=1)
        neighbors = np.take_along_axis(neighbors, order, axis=1)
        distances = np.take_along_axis(distances, order, axis=1)
        graph_neighbors[start:stop] = neighbors[:, :keep]
        graph_distances[start:stop] = distances[:, :keep]
        local_scales[start:stop] = distances[:, scale_keep - 1]

    positive_scales = local_scales[
        np.isfinite(local_scales) & (local_scales > 0)
    ]
    if positive_scales.size == 0:
        raise ValueError("No positive exact Fermat local scales were found.")
    local_scales[
        ~np.isfinite(local_scales) | (local_scales <= 0)
    ] = float(np.median(positive_scales))

    rows = np.repeat(np.arange(n_points), keep)
    columns = graph_neighbors.reshape(-1)
    edge_distances = graph_distances.reshape(-1)
    # Convert retained Fermat neighbors into the same self-tuned Gaussian
    # affinity used by the dense path.
    denominators = (
        setting["kernel_scale"] ** 2
        * local_scales[rows]
        * local_scales[columns]
    )
    denominators = np.maximum(denominators, np.finfo(float).eps)
    weights = np.exp(-(edge_distances**2 / denominators))
    weights[~np.isfinite(weights)] = 0.0
    directed = csr_matrix(
        (weights, (rows, columns)),
        shape=(n_points, n_points),
    )
    # Union symmetrization keeps an edge selected in either directed kNN row.
    affinity = directed.maximum(directed.T).tocsr()
    affinity.setdiag(0)
    affinity.eliminate_zeros()
    return affinity, local_scales


def _build_fall_graph(X, setting):
    # Build the rooted exact-Fermat graph used by fixed-p FALL.

    start = time.perf_counter()
    # Dense all-pairs distances are convenient for the paper-sized scenes.
    # The chunked route preserves the same construction for larger inputs.
    if X.shape[0] <= HARD_CODED_SETTINGS["exact_dense_limit"]:
        distances = fermat_distances(
            X,
            k=setting["fermat_k"],
            p=setting["p"],
        )
        affinity, _, local_scales = self_tuned_distance_affinity(
            distances,
            n_neighbors=setting["graph_neighbors"],
            scale_neighbors=HARD_CODED_SETTINGS[
                "self_tune_neighbors_k_sigma"
            ],
            scale=setting["kernel_scale"],
            chunk_size=HARD_CODED_SETTINGS["exact_chunk_size"],
        )
    else:
        affinity, local_scales = _chunked_exact_affinity(X, setting)
    seconds = time.perf_counter() - start
    _require_connected(affinity, "FALL graph")
    return {
        "affinity": affinity,
        "graph_seconds": seconds,
        "local_scale_median": float(np.median(local_scales)),
    }


def _build_afall_graphs(X, setting):
    # Precompute one Landmark-MDS affinity for every candidate exponent.

    # Landmark selection is deterministic and shared across candidate p values,
    # making candidate graphs comparable.
    landmark_start = time.perf_counter()
    landmarks = landmark_indices(
        X,
        count=min(setting["landmark_count"], X.shape[0]),
        strategy=HARD_CODED_SETTINGS["landmark_strategy"],
        random_state=HARD_CODED_SETTINGS["landmark_seed"],
    )
    landmark_seconds = time.perf_counter() - landmark_start

    candidates = {}
    errors = {}
    graph_seconds_total = 0.0
    for p in setting["candidate_p_tuple"]:
        start = time.perf_counter()
        try:
            distances = fermat_distances(
                X,
                k=setting["fermat_k"],
                p=p,
                indices=landmarks,
            )
            features = _landmark_mds_features(
                distances,
                landmarks=landmarks,
                n_components=setting["mds_dimension"],
            )
            affinity, _, local_scales = self_tuned_knn_affinity(
                features,
                n_neighbors=setting["graph_neighbors"],
                scale_neighbors=HARD_CODED_SETTINGS[
                    "self_tune_neighbors_k_sigma"
                ],
                scale=setting["kernel_scale"],
            )
            _require_connected(affinity, f"A-FALL candidate p={p:g}")
        # A numerically invalid or disconnected candidate is recorded and
        # skipped without discarding the other candidate graphs.
        except Exception as exc:  # noqa: BLE001
            errors[float(p)] = f"{type(exc).__name__}: {exc}"
            continue
        seconds = time.perf_counter() - start
        graph_seconds_total += seconds
        candidates[float(p)] = {
            "affinity": affinity,
            "graph_seconds": seconds,
            "local_scale_median": float(np.median(local_scales)),
        }

    if not candidates:
        raise ValueError(f"No A-FALL candidate graph succeeded: {errors}")
    return {
        "candidates": candidates,
        "landmarks": landmarks,
        "landmark_seconds": landmark_seconds,
        "graph_seconds_total": graph_seconds_total,
        "candidate_errors": errors,
    }


def _initial_label(ground_truth, seed):
    # Draw the single initial label uniformly from queryable foreground.

    rng = np.random.default_rng(seed)
    return int(rng.choice(np.flatnonzero(ground_truth != 0)))


def _solve(
    affinity,
    labeled,
    ground_truth,
    classes,
    setting,
):
    # Run one PWLL-tau classification solve for the current labeled set.

    labeled = np.asarray(labeled, dtype=int)
    # The query count starts at zero for the initial label.
    tau = pwll_tau_decay(
        setting["initial_tau"],
        query_iteration=labeled.size - 1,
        cluster_count=classes.size,
        epsilon=HARD_CODED_SETTINGS["tau_endpoint"],
    )
    options = _solver_options(setting["dataset"], setting["algorithm"])
    predictions, scores = poisson_reweighted_laplace_learning(
        affinity,
        labeled_indices=labeled,
        labeled_labels=ground_truth[labeled],
        classes=classes,
        reweight_tolerance=options["reweight_tolerance"],
        reweight_max_iterations=options["reweight_max_iterations"],
        gamma_floor=HARD_CODED_SETTINGS["gamma_floor"],
        tau=tau,
        poisson_gauge=options["poisson_gauge"],
    )
    return predictions, scores, tau


def _select_afall_p(bundle, current_p, labeled, ground_truth, classes, setting):
    # Score all valid A-FALL candidates and select the best exponent.

    options = _solver_options(setting["dataset"], "a-fall")
    labeled = np.asarray(labeled, dtype=int)
    labels = ground_truth[labeled]
    tau = pwll_tau_decay(
        setting["initial_tau"],
        query_iteration=labeled.size - 1,
        cluster_count=classes.size,
        epsilon=HARD_CODED_SETTINGS["tau_endpoint"],
    )

    scores = {}
    errors = {}
    validation_count = 0
    # ALOO freezes one full-label PWLL operator and uses Kron reduction;
    # ELOO reruns PWLL after holding out each eligible label.
    selector = (
        select_graph_by_approximate_leave_one_out
        if setting["p_selection_method"] == "aloo"
        else select_graph_by_leave_one_out
    )
    # Score candidates independently so one failed PWLL/LOO solve cannot
    # invalidate every p value at an update round.
    for p, candidate in bundle["candidates"].items():
        try:
            _, candidate_scores, count = (
                selector(
                    {float(p): candidate["affinity"]},
                    current_p=float(current_p),
                    labeled_indices=labeled,
                    labeled_labels=labels,
                    classes=classes,
                    min_validation_classes=HARD_CODED_SETTINGS[
                        "aloo_min_validation_classes"
                    ],
                    reweight_tolerance=options["reweight_tolerance"],
                    reweight_max_iterations=options[
                        "reweight_max_iterations"
                    ],
                    gamma_floor=HARD_CODED_SETTINGS["gamma_floor"],
                    poisson_gauge=options["poisson_gauge"],
                    tau=tau,
                    margin_weight=setting["loo_margin_weight"],
                )
            )
            scores.update(candidate_scores)
            validation_count = max(validation_count, count)
        except Exception as exc:  # noqa: BLE001
            errors[float(p)] = f"{type(exc).__name__}: {exc}"

    if not scores:
        return float(current_p), scores, errors, validation_count
    # Prefer the current exponent on score ties, then the smaller exponent.
    selected = max(
        sorted(scores),
        key=lambda p: (
            scores[p],
            -abs(p - float(current_p)),
            -p,
        ),
    )
    return float(selected), scores, errors, validation_count


def _row_base(setting, seed, budget):
    # Create the reproducibility fields shared by every raw result row.

    return {
        **{column: setting.get(column) for column in CONFIG_COLUMNS},
        "seed": int(seed),
        "budget": int(budget),
    }


def _run_seed(
    setting,
    graph_data,
    ground_truth,
    classes,
    seed,
):
    # Run one sequential active-learning trajectory for one random seed.

    # Background nodes remain in the graph but are never queried or scored.
    queryable = np.flatnonzero(ground_truth != 0)
    labeled = [_initial_label(ground_truth, seed)]
    current_p = setting.get("initial_p")
    cumulative_solve = 0.0
    cumulative_selection = 0.0
    rows = []
    final_artifact = None

    for budget in range(1, setting["max_budget"] + 1):
        selection_seconds = 0.0
        p_scores = {}
        p_errors = {}
        validation_count = 0
        p_updated = False

        if setting["algorithm"] == "a-fall":
            should_update = (
                budget >= setting["update_period"]
                and budget % setting["update_period"] == 0
                and budget < setting["max_budget"]
                and len(graph_data["candidates"]) > 1
            )
            if should_update:
                start = time.perf_counter()
                selected_p, p_scores, p_errors, validation_count = (
                    _select_afall_p(
                        graph_data,
                        current_p,
                        labeled,
                        ground_truth,
                        classes,
                        setting,
                    )
                )
                selection_seconds = time.perf_counter() - start
                cumulative_selection += selection_seconds
                if selected_p in graph_data["candidates"] and not np.isclose(
                    selected_p,
                    current_p,
                ):
                    current_p = selected_p
                    p_updated = True
            # Selection occurs before the PWLL solve, so a newly chosen p takes
            # effect immediately at this budget.
            affinity = graph_data["candidates"][float(current_p)]["affinity"]
        else:
            affinity = graph_data["affinity"]

        start = time.perf_counter()
        try:
            predictions, scores, tau = _solve(
                affinity,
                labeled,
                ground_truth,
                classes,
                setting,
            )
            error = ""
        except Exception as exc:  # noqa: BLE001
            predictions = None
            scores = None
            tau = np.nan
            error = f"{type(exc).__name__}: {exc}"
        solve_seconds = time.perf_counter() - start
        cumulative_solve += solve_seconds

        labeled_array = np.asarray(labeled, dtype=int)
        labeled_classes = np.unique(ground_truth[labeled_array])
        covered = int(labeled_classes.size)
        if setting["algorithm"] == "a-fall":
            graph_overhead = (
                graph_data["landmark_seconds"]
                + graph_data["graph_seconds_total"]
            )
        else:
            graph_overhead = graph_data["graph_seconds"]
        # Per-run runtime includes all candidate graph construction, all PWLL
        # solves up to this budget, and every completed p-selection update.
        runtime = graph_overhead + cumulative_solve + cumulative_selection

        row = {
            **_row_base(setting, seed, budget),
            "OA": (
                np.nan
                if predictions is None
                else accuracy(predictions, ground_truth)
            ),
            "AA": (
                np.nan
                if predictions is None
                else average_accuracy(predictions, ground_truth, classes)
            ),
            "covered_classes": covered,
            "full_coverage": float(covered == classes.size),
            "active_p": (
                float(current_p)
                if setting["algorithm"] == "a-fall"
                else setting.get("p")
            ),
            "p_updated": float(p_updated),
            "p_scores": json.dumps(p_scores, sort_keys=True),
            "p_errors": json.dumps(p_errors, sort_keys=True),
            "p_validation_count": int(validation_count),
            "tau": tau,
            "labeled_indices": ";".join(
                str(int(index)) for index in labeled_array
            ),
            "labeled_classes": ";".join(
                str(int(label)) for label in sorted(labeled_classes)
            ),
            "solve_seconds": solve_seconds,
            "cumulative_solve_seconds": cumulative_solve,
            "selection_seconds": selection_seconds,
            "cumulative_selection_seconds": cumulative_selection,
            "graph_overhead_seconds": graph_overhead,
            "runtime_seconds": runtime,
            "error": error,
        }
        rows.append(row)

        if error:
            break
        if budget == setting["max_budget"]:
            final_artifact = {
                "seed": int(seed),
                "predictions": predictions.copy(),
                "labeled_indices": labeled_array.copy(),
                "OA": row["OA"],
                "AA": row["AA"],
                "runtime_seconds": runtime,
                "active_p": row["active_p"],
            }
            break

        labeled_mask = np.zeros(ground_truth.size, dtype=bool)
        labeled_mask[labeled_array] = True
        candidates = queryable[~labeled_mask[queryable]]
        if candidates.size == 0:
            break
        # ``minimum_norm_acquisition`` returns larger values for lower score
        # norms, hence argmax chooses the most uncertain queryable pixel.
        acquisition = minimum_norm_acquisition(scores, candidates)
        labeled.append(int(candidates[np.argmax(acquisition)]))

    return rows, final_artifact


def _best_artifact(artifacts):
    # Choose the best final seed by OA, then AA, then lower seed number.

    successful = [item for item in artifacts if item is not None]
    if not successful:
        return None
    return max(
        successful,
        key=lambda item: (item["OA"], item["AA"], -item["seed"]),
    )


def _encode_labels(values, classes):
    # Map dataset class IDs to contiguous indices for plotting.

    encoded = np.zeros(values.shape, dtype=int)
    for position, label in enumerate(classes, start=1):
        encoded[values == label] = position
    return encoded


def save_best_visualization(
    output_path,
    setting,
    artifact,
    ground_truth,
    spatial_shape,
    classes,
):
    # Save GT, best prediction, queried points, and their raw arrays.

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    color_values = [(0.0, 0.0, 0.0, 1.0)]
    color_values.extend(plt.get_cmap("tab20")(np.linspace(0, 1, classes.size)))
    cmap = ListedColormap(color_values)
    norm = BoundaryNorm(
        np.arange(classes.size + 2) - 0.5,
        cmap.N,
    )

    gt_encoded = _encode_labels(ground_truth, classes).reshape(spatial_shape)
    # Predictions exist for all graph nodes, but background is blacked out to
    # match the foreground-only OA/AA evaluation protocol.
    displayed = artifact["predictions"].copy()
    displayed[ground_truth == 0] = 0
    pred_encoded = _encode_labels(displayed, classes).reshape(spatial_shape)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        constrained_layout=True,
    )
    axes[0].imshow(gt_encoded, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("Ground truth")
    axes[1].imshow(pred_encoded, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title(
        f"Best seed {artifact['seed']}\n"
        f"OA={artifact['OA']:.4f}, AA={artifact['AA']:.4f}"
    )
    axes[2].imshow(
        gt_encoded,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=0.35,
    )
    query_rows, query_cols = np.unravel_index(
        artifact["labeled_indices"],
        spatial_shape,
    )
    query_colors = [
        color_values[
            int(np.flatnonzero(classes == ground_truth[index])[0]) + 1
        ]
        for index in artifact["labeled_indices"]
    ]
    axes[2].scatter(
        query_cols,
        query_rows,
        c=query_colors,
        edgecolors="white",
        linewidths=0.5,
        s=24,
    )
    axes[2].set_title(f"Queried pixels, B={setting['max_budget']}")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(setting["setting_id"], fontsize=9)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)

    np.savez_compressed(
        output_path.with_suffix(".npz"),
        predictions=artifact["predictions"],
        labeled_indices=artifact["labeled_indices"],
        seed=artifact["seed"],
        OA=artifact["OA"],
        AA=artifact["AA"],
        runtime_seconds=artifact["runtime_seconds"],
        active_p=artifact["active_p"],
    )


def summarize_results(rows, settings):
    # Aggregate final-budget results with the submitted-table conventions.

    frame = pd.DataFrame(rows)
    summary_rows = []
    setting_lookup = {setting["setting_id"]: setting for setting in settings}
    for setting_id, setting in setting_lookup.items():
        group = frame[
            (frame["setting_id"] == setting_id)
            & (frame["budget"] == setting["max_budget"])
            & (frame["error"].fillna("") == "")
        ].copy()
        if group.empty:
            continue
        # Preserve the dispersion convention used for the submitted table:
        # population SD for the released PWLL baseline and sample SD otherwise.
        ddof = 0 if setting["algorithm"] == "pwll-tau" else 1
        if group.shape[0] <= ddof:
            oa_std = 0.0
            aa_std = 0.0
            runtime_std = 0.0
        else:
            oa_std = float(group["OA"].std(ddof=ddof))
            aa_std = float(group["AA"].std(ddof=ddof))
            runtime_std = float(group["runtime_seconds"].std(ddof=ddof))
        best = group.sort_values(
            ["OA", "AA", "seed"],
            ascending=[False, False, True],
        ).iloc[0]
        summary_rows.append(
            {
                **{column: setting.get(column) for column in CONFIG_COLUMNS},
                "completed_seeds": int(group["seed"].nunique()),
                "OA_mean": float(group["OA"].mean()),
                "OA_std": oa_std,
                "AA_mean": float(group["AA"].mean()),
                "AA_std": aa_std,
                "covered_classes_mean": float(
                    group["covered_classes"].mean()
                ),
                "runtime_mean_seconds": float(
                    group["runtime_seconds"].mean()
                ),
                "runtime_std_seconds": runtime_std,
                "best_seed": int(best["seed"]),
                "best_seed_OA": float(best["OA"]),
                "best_seed_AA": float(best["AA"]),
            }
        )
    return pd.DataFrame(summary_rows)


def _write_outputs(output_dir, rows, summary):
    # Write the complete per-budget trace and final-budget summary.

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "raw_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)


def save_budget_curves(output_dir, rows, settings):
    # Save mean OA/AA trajectories for every completed configuration.

    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame = frame[frame["error"].fillna("").eq("")].copy()
    if frame.empty:
        return

    curve_parts = []
    for setting in settings:
        group = frame[frame["setting_id"] == setting["setting_id"]]
        if group.empty:
            continue
        curve = (
            group.groupby("budget", sort=True)
            .agg(
                OA_mean=("OA", "mean"),
                OA_std=("OA", "std"),
                AA_mean=("AA", "mean"),
                AA_std=("AA", "std"),
                completed_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        curve[["OA_std", "AA_std"]] = curve[
            ["OA_std", "AA_std"]
        ].fillna(0.0)
        curve["algorithm"] = setting["algorithm"]
        curve["setting_id"] = setting["setting_id"]
        curve["p_selection_method"] = setting.get("p_selection_method")
        curve_parts.append(curve)

    if not curve_parts:
        return
    curves = pd.concat(curve_parts, ignore_index=True)
    output_dir = Path(output_dir)
    curves.to_csv(output_dir / "budget_curves.csv", index=False)

    for metric in ("OA", "AA"):
        figure, axis = plt.subplots(
            figsize=(8.0, 5.5),
            constrained_layout=True,
        )
        for setting in settings:
            curve = curves[curves["setting_id"] == setting["setting_id"]]
            if curve.empty:
                continue
            method = setting.get("p_selection_method")
            if setting["algorithm"] == "a-fall" and method:
                label = f"A-FALL ({method.upper()})"
            else:
                label = setting["algorithm"].upper()
            if sum(item["algorithm"] == setting["algorithm"] for item in settings) > 1:
                if setting["algorithm"] != "a-fall" or not method:
                    label = setting["setting_id"]

            budget = curve["budget"].to_numpy(dtype=float)
            mean = curve[f"{metric}_mean"].to_numpy(dtype=float)
            std = curve[f"{metric}_std"].to_numpy(dtype=float)
            line = axis.plot(
                budget,
                mean,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=label,
            )[0]
            axis.fill_between(
                budget,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=line.get_color(),
                alpha=0.14,
                linewidth=0,
            )

        axis.set_xlabel("Label budget B")
        axis.set_ylabel(metric)
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        figure.savefig(output_dir / f"{metric.lower()}_curve.png", dpi=220)
        plt.close(figure)


def _json_ready(value):
    # Recursively convert NumPy values into JSON-serializable Python types.

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_metadata(output_dir, args, settings, data_info, wall_seconds):
    # Save command, configuration, data shape, timing, and package versions.

    metadata = {
        "command": sys.argv,
        "dataset": args.dataset,
        "algorithms": args.algorithms,
        "requested_seeds": args.seeds,
        "seed_start": args.seed_start,
        "paper_preset": asdict(PAPER_PRESETS[args.dataset]),
        "hard_coded_settings": HARD_CODED_SETTINGS,
        "settings": settings,
        "data": data_info,
        "wall_seconds": wall_seconds,
        "platform": {
            "python": sys.version,
            "system": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    path = Path(output_dir) / "metadata.json"
    path.write_text(
        json.dumps(_json_ready(metadata), indent=2, sort_keys=True),
        encoding="ascii",
    )


def _print_setting(setting, position, total):
    # Print one expanded configuration before it starts.

    displayed = {
        key: setting.get(key)
        for key in CONFIG_COLUMNS
        if setting.get(key) not in (None, "")
    }
    print(
        f"\n[{position}/{total}] {setting['setting_id']}",
        flush=True,
    )
    print(
        "  hyperparameters:",
        json.dumps(displayed, sort_keys=True),
        flush=True,
    )


def run_experiment_grid(args):
    # Run every requested algorithm/grid configuration end to end.

    settings = build_settings(args)
    if not settings:
        raise ValueError("The requested options produced no experiment settings.")

    print("dataset:", args.dataset, flush=True)
    print("algorithms:", ", ".join(args.algorithms), flush=True)
    print("seeds:", args.seeds, flush=True)
    print("grid configurations:", len(settings), flush=True)
    for index, setting in enumerate(settings, start=1):
        print(f"  {index}: {setting['setting_id']}", flush=True)
    if args.dry_run:
        return pd.DataFrame()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. Pass --overwrite or choose another "
            "output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"

    total_start = time.perf_counter()
    X, ground_truth, spatial_shape = load_hsi_dataset(
        args.dataset,
        dataset_dir=args.dataset_dir,
        return_shape=True,
    )
    classes = np.unique(ground_truth[ground_truth != 0])
    data_info = {
        "X_shape": X.shape,
        "spatial_shape": spatial_shape,
        "foreground_pixels": int(np.count_nonzero(ground_truth != 0)),
        "classes": classes.tolist(),
    }
    print("X shape:", X.shape, flush=True)
    print("spatial shape:", spatial_shape, flush=True)
    print("classes:", classes.tolist(), flush=True)

    # Graphs are deterministic for a configuration and dominate runtime, so
    # settings that differ only in budget, tau, or seed reuse graph data.
    graph_cache = {}
    all_rows = []
    for position, setting in enumerate(settings, start=1):
        _print_setting(setting, position, len(settings))
        algorithm = setting["algorithm"]
        if algorithm == "pwll-tau":
            graph_key = (
                algorithm,
                setting["pwll_k"],
                setting["gaussian_coefficient"],
            )
            builder = _build_pwll_graph
        elif algorithm == "fall":
            graph_key = (
                algorithm,
                setting["p"],
                setting["fermat_k"],
                setting["graph_neighbors"],
                setting["kernel_scale"],
            )
            builder = _build_fall_graph
        else:
            graph_key = (
                algorithm,
                setting["candidate_p_tuple"],
                setting["fermat_k"],
                setting["graph_neighbors"],
                setting["kernel_scale"],
                setting["landmark_count"],
                setting["mds_dimension"],
            )
            builder = _build_afall_graphs

        if graph_key not in graph_cache:
            print("  building graph data...", flush=True)
            try:
                graph_cache[graph_key] = builder(X, setting)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  skipped configuration: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
        else:
            print("  reusing cached graph data.", flush=True)
        graph_data = graph_cache[graph_key]

        artifacts = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            rows, artifact = _run_seed(
                setting,
                graph_data,
                ground_truth,
                classes,
                seed,
            )
            all_rows.extend(rows)
            # Persist after each seed so a long sweep retains completed work if
            # a later solver fails or the process is interrupted.
            pd.DataFrame(all_rows).to_csv(
                output_dir / "raw_results.csv",
                index=False,
            )
            artifacts.append(artifact)
            if artifact is None:
                error = rows[-1]["error"] if rows else "no result"
                print(f"  seed={seed}: FAILED {error}", flush=True)
            else:
                print(
                    f"  seed={seed}: OA={artifact['OA']:.6f}, "
                    f"AA={artifact['AA']:.6f}, "
                    f"runtime={artifact['runtime_seconds']:.2f}s, "
                    f"active_p={artifact['active_p']}",
                    flush=True,
                )

        summary = summarize_results(all_rows, settings)
        _write_outputs(output_dir, all_rows, summary)

        setting_summary = summary[
            summary["setting_id"] == setting["setting_id"]
        ]
        if not setting_summary.empty:
            result = setting_summary.iloc[0]
            print(
                "  aggregate:",
                f"OA={result['OA_mean']:.6f}+/-{result['OA_std']:.6f},",
                f"AA={result['AA_mean']:.6f}+/-{result['AA_std']:.6f},",
                f"runtime={result['runtime_mean_seconds']:.2f}s,",
                f"seeds={int(result['completed_seeds'])}",
                flush=True,
            )

        best = _best_artifact(artifacts)
        if best is not None:
            figure_path = visualization_dir / (
                f"{setting['setting_id']}__best_seed{best['seed']}.png"
            )
            save_best_visualization(
                figure_path,
                setting,
                best,
                ground_truth,
                spatial_shape,
                classes,
            )
            print("  saved best-seed visualization:", figure_path, flush=True)

    summary = summarize_results(all_rows, settings)
    _write_outputs(output_dir, all_rows, summary)
    save_budget_curves(output_dir, all_rows, settings)
    wall_seconds = time.perf_counter() - total_start
    _write_metadata(output_dir, args, settings, data_info, wall_seconds)

    print("\nFinal grid summary", flush=True)
    if summary.empty:
        print("  No configurations completed successfully.", flush=True)
    else:
        columns = [
            "algorithm",
            "setting_id",
            "completed_seeds",
            "OA_mean",
            "OA_std",
            "AA_mean",
            "AA_std",
            "runtime_mean_seconds",
            "best_seed",
        ]
        print(summary[columns].round(6).to_string(index=False), flush=True)
    print("saved:", output_dir / "raw_results.csv", flush=True)
    print("saved:", output_dir / "summary.csv", flush=True)
    print("saved:", output_dir / "budget_curves.csv", flush=True)
    print("saved:", output_dir / "oa_curve.png", flush=True)
    print("saved:", output_dir / "aa_curve.png", flush=True)
    print("saved:", output_dir / "metadata.json", flush=True)
    print("wall seconds:", wall_seconds, flush=True)
    return summary
