# Command-line entry point for the reproducible paper experiments.
#
# The parser accepts either one value or a list for each exposed
# hyperparameter. ``BackEnd.experiment_runner`` expands those values into a
# Cartesian grid and runs every configuration over the requested seeds.

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
# Allow this file to be launched directly from either the workspace root or
# inside ``Fermat_repo`` without installing BackEnd as a package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BackEnd.experiment_runner import (  # noqa: E402
    HARD_CODED_SETTINGS,
    PAPER_PRESETS,
    run_experiment_grid,
)


def _flatten(values):
    # Flatten comma- or space-separated CLI values into plain strings.

    if values is None:
        return []
    flattened = []
    for value in values:
        value = str(value).strip().strip("[]()")
        flattened.extend(
            item.strip().strip("[]()")
            for item in value.split(",")
            if item.strip().strip("[]()")
        )
    return flattened


def _numbers(values, cast, default):
    # Parse a numeric list, falling back to a copied default sequence.

    raw = _flatten(values)
    return [cast(value) for value in raw] if raw else list(default)


def _algorithms(values):
    # Validate algorithm names and convert aliases to canonical names.

    aliases = {
        "pwll": "pwll-tau",
        "pwll-tau": "pwll-tau",
        "fall": "fall",
        "a-fall": "a-fall",
        "afall": "a-fall",
    }
    requested = [value.lower() for value in _flatten(values or ["all"])]
    if "all" in requested:
        return ["pwll-tau", "fall", "a-fall"]

    result = []
    for value in requested:
        if value not in aliases:
            valid = "all, pwll-tau, fall, a-fall"
            raise ValueError(f"Unknown algorithm '{value}'. Expected: {valid}.")
        canonical = aliases[value]
        if canonical not in result:
            result.append(canonical)
    return result


def _p_selection_methods(values):
    # Normalize approximate/exact LOO aliases used by A-FALL.

    aliases = {
        "aloo": "aloo",
        "aloocv": "aloo",
        "eloo": "eloo",
        "exactloo": "eloo",
    }
    result = []
    for value in (item.lower() for item in _flatten(values or ["aloo"])):
        if value not in aliases:
            raise ValueError(
                "Unknown p-selection method "
                f"'{value}'. Expected aloo or eloo."
            )
        canonical = aliases[value]
        if canonical not in result:
            result.append(canonical)
    return result


def _positive(values, name, allow_zero=False):
    # Apply the shared positivity checks used by numeric grid arguments.

    lower_bound = 0 if allow_zero else 0
    for value in values:
        invalid = value < lower_bound if allow_zero else value <= lower_bound
        if invalid:
            relation = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} values must be {relation}.")


def _resolve_args(args):
    # Fill dataset presets, parse grids, and enforce cross-argument rules.

    preset = PAPER_PRESETS[args.dataset]
    args.algorithms = _algorithms(args.algorithms)
    args.budget_values = _numbers(
        args.budget_values,
        int,
        [preset.budget],
    )
    args.initial_tau_values = _numbers(
        args.initial_tau_values,
        float,
        [1e-3],
    )
    args.pwll_k_values = _numbers(args.pwll_k_values, int, [20])
    args.pwll_gaussian_coefficient_values = _numbers(
        args.pwll_gaussian_coefficient_values,
        float,
        [4.0],
    )
    args.fall_p_values = _numbers(
        args.fall_p_values,
        float,
        [preset.fall_p],
    )
    args.fermat_k_values = _numbers(args.fermat_k_values, int, [20])
    args.graph_neighbor_values = _numbers(
        args.graph_neighbor_values,
        int,
        [20],
    )
    args.kernel_scale_values = _numbers(
        args.kernel_scale_values,
        float,
        [8.0],
    )
    args.candidate_p_values = _numbers(
        args.candidate_p_values,
        float,
        preset.afall_candidate_p,
    )
    args.afall_initial_p_values = _numbers(
        args.afall_initial_p_values,
        float,
        [preset.afall_initial_p],
    )
    args.landmark_count_values = _numbers(
        args.landmark_count_values,
        int,
        [300],
    )
    args.mds_dimension_values = _numbers(
        args.mds_dimension_values,
        int,
        [32],
    )
    args.update_period_values = _numbers(
        args.update_period_values,
        int,
        [10],
    )
    args.loo_margin_weight_values = _numbers(
        args.loo_margin_weight_values,
        float,
        [HARD_CODED_SETTINGS["loo_margin_weight_default"]],
    )
    args.p_selection_methods = _p_selection_methods(
        args.p_selection_methods
    )

    _positive(args.budget_values, "budget")
    _positive(args.initial_tau_values, "initial tau", allow_zero=True)
    _positive(args.pwll_k_values, "PWLL k")
    _positive(
        args.pwll_gaussian_coefficient_values,
        "PWLL Gaussian coefficient",
    )
    _positive(args.fall_p_values, "FALL p")
    _positive(args.fermat_k_values, "Fermat k")
    _positive(args.graph_neighbor_values, "graph neighbor")
    _positive(args.kernel_scale_values, "kernel scale")
    _positive(args.candidate_p_values, "candidate p")
    _positive(args.afall_initial_p_values, "A-FALL initial p")
    _positive(args.landmark_count_values, "landmark count")
    _positive(args.mds_dimension_values, "MDS dimension")
    _positive(args.update_period_values, "update period")
    _positive(
        args.loo_margin_weight_values,
        "LOO relative margin weight",
        allow_zero=True,
    )
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive.")

    # A-FALL must begin on a graph that was actually precomputed as part of
    # its candidate family.
    candidate_set = set(args.candidate_p_values)
    missing = [
        value
        for value in args.afall_initial_p_values
        if value not in candidate_set
    ]
    if "a-fall" in args.algorithms and missing:
        raise ValueError(
            "Every --afall-initial-p-values entry must also appear in "
            f"--candidate-p-values. Missing: {missing}"
        )

    if args.output_dir is None:
        # Keep independent dataset runs separate when the user does not choose
        # an explicit destination.
        args.output_dir = str(
            ROOT / "runs" / args.dataset
        )
    return args


def build_parser():
    # Define the public experiment CLI and its paper-facing parameters.

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run the paper's PWLL-tau, exact-Fermat FALL, and LMDS A-FALL "
            "experiments with ALOO or ELOO p-selection. Every "
            "parameter-grid flag accepts either one value or several "
            "comma/space-separated values. Multiple values are expanded "
            "as a Cartesian grid."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(PAPER_PRESETS),
        required=True,
        help="SalinasA or the fixed 75x180 PaviaU subset.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        help="Any of: all, pwll-tau, fall, a-fall.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "datasets"),
        help="Directory containing the four required .mat files.",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--budget-values",
        nargs="+",
        help="Label budgets B. Defaults to 20 (SalinasA) or 30 (PaviaU).",
    )
    parser.add_argument(
        "--initial-tau-values",
        nargs="+",
        help="Initial PWLL diagonal perturbations tau_0.",
    )

    pwll = parser.add_argument_group("PWLL-tau parameters")
    pwll.add_argument(
        "--pwll-k-values",
        nargs="+",
        help="Released PWLL angular-kNN graph sizes.",
    )
    pwll.add_argument(
        "--pwll-gaussian-coefficient-values",
        nargs="+",
        help="Coefficients in exp(-coefficient*d_ij^2/d_k(i)^2).",
    )

    fall = parser.add_argument_group("FALL parameters")
    fall.add_argument(
        "--fall-p-values",
        nargs="+",
        help="Fixed exact-Fermat exponents p.",
    )

    fermat = parser.add_argument_group("Shared FALL/A-FALL graph parameters")
    fermat.add_argument(
        "--fermat-k-values",
        nargs="+",
        help="k_F values for the sparse powered-distance graph.",
    )
    fermat.add_argument(
        "--graph-neighbor-values",
        nargs="+",
        help="k_G values retained in the propagation graph.",
    )
    fermat.add_argument(
        "--kernel-scale-values",
        nargs="+",
        help="Self-tuned kernel scales eta.",
    )

    afall = parser.add_argument_group("A-FALL parameters")
    afall.add_argument(
        "--candidate-p-values",
        nargs="+",
        help=(
            "One candidate set P used inside every A-FALL run. Unlike the "
            "other list flags, this list is not itself expanded as a grid."
        ),
    )
    afall.add_argument(
        "--afall-initial-p-values",
        nargs="+",
        help="Initial exponents p_0; multiple values form a grid.",
    )
    afall.add_argument(
        "--landmark-count-values",
        nargs="+",
        help="Numbers m of FPS landmarks.",
    )
    afall.add_argument(
        "--mds-dimension-values",
        nargs="+",
        help="LMDS embedding dimensions r.",
    )
    afall.add_argument(
        "--update-period-values",
        nargs="+",
        help="ALOO p-update periods T.",
    )
    afall.add_argument(
        "--loo-margin-weight-values",
        "--aloo-margin-weight-values",
        dest="loo_margin_weight_values",
        nargs="+",
        help=(
            "Relative ALOO/ELOO margin weights lambda_2/lambda_1; the "
            "loss coefficient is normalized to one."
        ),
    )
    afall.add_argument(
        "--p-selection-method",
        dest="p_selection_methods",
        nargs="+",
        default=["aloo"],
        help=(
            "A-FALL p-selection method: aloo (approximate, default) or "
            "eloo (exact leave-one-out). Supplying both runs both methods."
        ),
    )

    output = parser.add_argument_group("Output and control")
    output.add_argument(
        "--output-dir",
        help="Directory for raw CSV, summary, metadata, and visualizations.",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a nonempty output directory.",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded grid without loading data or running it.",
    )
    return parser


def main():
    # Parse, validate, expand, and execute the requested experiment grid.

    args = _resolve_args(build_parser().parse_args())
    run_experiment_grid(args)


if __name__ == "__main__":
    main()
