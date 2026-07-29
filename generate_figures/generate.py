# Generate publication figures from saved experiment outputs.
#
# The utility accepts a run directory, raw per-seed CSV, or aggregated curve
# CSV. Raw results can produce both budget curves and best-seed maps when their
# companion ``.npz`` artifacts are available; aggregate CSVs contain enough
# information for curves only.

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
FERMAT_REPO = PACKAGE_DIR.parent
WORKSPACE_ROOT = FERMAT_REPO.parent
if str(FERMAT_REPO) not in sys.path:
    sys.path.insert(0, str(FERMAT_REPO))

from BackEnd.preprocessing import load_hsi_dataset  # noqa: E402


RAW_METRICS = ("OA", "AA")
MEAN_METRICS = ("OA_mean", "AA_mean")
AUTO_GROUP_COLUMNS = ("setting_id", "name", "method", "algorithm")


def _safe_name(value):
    # Convert an arbitrary setting label into a portable filename.

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_") or "result"


def _resolve_source(input_path):
    # Resolve a run directory or explicit CSV to the result table to read.

    input_path = Path(input_path).resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError("The input file must be a CSV file.")
        return input_path, input_path.parent

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    # New reproducibility runs use these canonical names. Legacy ``*_raw``
    # files remain supported only when the choice is unambiguous.
    preferred = (
        input_path / "raw_results.csv",
        input_path / "budget_curves.csv",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate, input_path

    raw_candidates = sorted(input_path.glob("*_raw.csv"))
    if len(raw_candidates) == 1:
        return raw_candidates[0], input_path
    if not raw_candidates:
        raise FileNotFoundError(
            f"No raw_results.csv, budget_curves.csv, or *_raw.csv was "
            f"found in {input_path}."
        )
    names = "\n  ".join(path.name for path in raw_candidates)
    raise ValueError(
        "The input directory contains multiple legacy raw CSV files. Pass "
        f"one CSV explicitly:\n  {names}"
    )


def _default_output_dir(input_path, source_path):
    # Choose a figure directory beside the supplied run or CSV.

    input_path = Path(input_path).resolve()
    if input_path.is_dir():
        return input_path / "figures"
    return source_path.parent / f"{source_path.stem}_figures"


def _load_metadata(run_dir, source_path):
    # Load optional run metadata used to recover the dataset identity.

    candidates = (
        Path(run_dir) / "metadata.json",
        source_path.parent / "metadata.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8")), candidate
    return {}, None


def _successful_rows(frame):
    # Remove rows whose solver or graph construction reported an error.

    frame = frame.copy()
    if "error" in frame.columns:
        errors = frame["error"].fillna("").astype(str).str.strip()
        frame = frame[errors.eq("")].copy()
    return frame


def _detect_group_columns(frame, requested):
    # Find columns that identify independent curves/configurations.

    if requested:
        missing = [column for column in requested if column not in frame.columns]
        if missing:
            raise ValueError(
                "Requested group columns are absent from the CSV: "
                + ", ".join(missing)
            )
        return list(requested)

    for column in AUTO_GROUP_COLUMNS:
        if column in frame.columns:
            return [column]
    return []


def _add_group_key(frame, group_columns, fallback):
    # Create one stable internal grouping key from one or more columns.

    frame = frame.copy()
    if not group_columns:
        frame["_figure_group"] = fallback
        return frame

    values = frame[group_columns].fillna("").astype(str)
    frame["_figure_group"] = values.apply(
        lambda row: " | ".join(
            f"{column}={row[column]}" if len(group_columns) > 1 else row[column]
            for column in group_columns
        ),
        axis=1,
    )
    return frame


def _format_number(value):
    # Format optional numeric hyperparameters compactly for legends.

    if value is None or pd.isna(value):
        return None
    return f"{float(value):g}"


def _display_label(group, group_frame):
    # Build a concise, paper-readable legend label for one setting.

    first = group_frame.iloc[0]
    if "setting_id" not in group_frame.columns:
        return str(group)

    algorithm = str(first.get("algorithm", "")).lower()
    if algorithm == "pwll-tau":
        return r"PWLL-$\tau$"
    if algorithm == "fall":
        p_value = _format_number(first.get("p"))
        return "FALL" if p_value is None else f"FALL (p={p_value})"
    if algorithm == "a-fall":
        method = str(first.get("p_selection_method", "")).upper()
        initial_p = _format_number(first.get("initial_p"))
        details = [item for item in (method, f"p0={initial_p}" if initial_p else "") if item]
        return "A-FALL" if not details else f"A-FALL ({', '.join(details)})"
    return str(group)


def _make_labels(frame):
    # Create unique display labels, falling back to full keys on collisions.

    labels = {}
    for group, group_frame in frame.groupby("_figure_group", sort=False):
        labels[group] = _display_label(group, group_frame)

    counts = pd.Series(list(labels.values())).value_counts()
    for group, label in list(labels.items()):
        if counts[label] > 1:
            labels[group] = str(group)
    return labels


def _aggregate_raw(frame):
    # Aggregate per-seed OA/AA into mean and standard-deviation curves.

    required = {"budget", "OA", "AA"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "A raw result CSV must contain: budget, OA, and AA. Missing: "
            + ", ".join(sorted(missing))
        )

    frame = frame.copy()
    for column in ("budget", "OA", "AA"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["budget", "OA", "AA"])
    if frame.empty:
        raise ValueError("No successful numeric OA/AA rows remain.")

    if "seed" not in frame.columns:
        frame["seed"] = 0

    curves = (
        frame.groupby(["_figure_group", "budget"], sort=True)
        .agg(
            OA_mean=("OA", "mean"),
            OA_std=("OA", "std"),
            AA_mean=("AA", "mean"),
            AA_std=("AA", "std"),
            completed_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    curves[["OA_std", "AA_std"]] = curves[
        ["OA_std", "AA_std"]
    ].fillna(0.0)
    return curves


def _normalize_aggregated(frame):
    # Map an already aggregated CSV onto the internal curve schema.

    required = {"budget", *MEAN_METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "An aggregated result CSV must contain budget, OA_mean, and "
            "AA_mean. Missing: " + ", ".join(sorted(missing))
        )

    curves = frame.copy()
    for metric in RAW_METRICS:
        std_column = f"{metric}_std"
        if std_column not in curves.columns:
            curves[std_column] = 0.0
    if "completed_seeds" not in curves.columns:
        curves["completed_seeds"] = np.nan
    return curves[
        [
            "_figure_group",
            "budget",
            "OA_mean",
            "OA_std",
            "AA_mean",
            "AA_std",
            "completed_seeds",
        ]
    ].copy()


def build_curves(frame, source_path, group_columns):
    # Prepare grouped labels and normalized OA/AA budget trajectories.

    frame = _successful_rows(frame)
    frame = _add_group_key(frame, group_columns, source_path.stem)
    labels = _make_labels(frame)
    if set(RAW_METRICS).issubset(frame.columns):
        curves = _aggregate_raw(frame)
    elif set(MEAN_METRICS).issubset(frame.columns):
        curves = _normalize_aggregated(frame)
    else:
        raise ValueError(
            "The CSV must contain either OA and AA, or OA_mean and AA_mean."
        )
    curves["label"] = curves["_figure_group"].map(labels)
    return curves, frame, labels


def _plot_metric(curves, metric, output_path, title, dpi):
    # Plot a mean budget curve with a clipped one-standard-deviation band.

    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    groups = list(curves["_figure_group"].drop_duplicates())
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(groups), 1)))

    for color, group in zip(colors, groups):
        curve = curves[curves["_figure_group"] == group].sort_values("budget")
        budget = curve["budget"].to_numpy(dtype=float)
        mean = curve[f"{metric}_mean"].to_numpy(dtype=float)
        std = curve[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        label = str(curve["label"].iloc[0])
        axis.plot(
            budget,
            mean,
            color=color,
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            label=label,
        )
        axis.fill_between(
            budget,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    axis.set_xlabel(r"Label budget $B$")
    axis.set_ylabel(metric)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, alpha=0.25)
    if title:
        axis.set_title(title)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_curves(curves, output_dir, title, dpi):
    # Save the normalized curve CSV and separate OA/AA figures.

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "budget_curves.csv"
    curves.to_csv(curve_path, index=False)
    outputs = [curve_path]
    for metric in RAW_METRICS:
        output_path = output_dir / f"{metric.lower()}_curve.png"
        _plot_metric(curves, metric, output_path, title, dpi)
        outputs.append(output_path)
    return outputs


def _infer_dataset(frame, metadata, requested):
    # Determine the dataset from an override, metadata, or result column.

    if requested:
        return requested
    if metadata.get("dataset"):
        return str(metadata["dataset"])
    if "dataset" in frame.columns:
        values = frame["dataset"].dropna().astype(str).unique()
        if values.size == 1:
            return str(values[0])
    return None


def _artifact_candidates(run_dir, source_path, requested):
    # Collect unique prediction artifacts from standard run locations.

    directories = []
    if requested:
        directories.append(Path(requested).resolve())
    directories.extend(
        [
            Path(run_dir) / "visualizations",
            source_path.parent / "visualizations",
            Path(run_dir),
            source_path.parent,
        ]
    )
    artifacts = []
    seen = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for artifact in sorted(directory.glob("*.npz")):
            resolved = artifact.resolve()
            if resolved not in seen:
                artifacts.append(resolved)
                seen.add(resolved)
    return artifacts


def _select_best_rows(frame):
    # Select the highest-OA seed at each setting's final budget.
    #
    # AA is the first tie-breaker and the smaller seed is the final deterministic
    # tie-breaker, matching the experiment runner.

    if not {"budget", "OA", "AA"}.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    for column in ("budget", "OA", "AA"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["budget", "OA", "AA"])
    if frame.empty:
        return frame
    if "seed" not in frame.columns:
        frame["seed"] = 0

    selected = []
    for _, group in frame.groupby("_figure_group", sort=False):
        final = group[group["budget"] == group["budget"].max()]
        selected.append(
            final.sort_values(
                ["OA", "AA", "seed"],
                ascending=[False, False, True],
            ).iloc[0]
        )
    return pd.DataFrame(selected)


def _read_artifact_header(path):
    # Read only the scalar fields needed to identify an ``.npz`` artifact.

    with np.load(path, allow_pickle=False) as artifact:
        return {
            "seed": int(np.asarray(artifact["seed"]).item()),
            "OA": float(np.asarray(artifact["OA"]).item()),
            "AA": float(np.asarray(artifact["AA"]).item()),
        }


def _match_artifact(row, artifacts):
    # Match a result row to its exact artifact or an unambiguous fallback.

    seed = int(row.get("seed", 0))
    setting_id = str(row.get("setting_id", ""))
    exact_name = (
        f"{_safe_name(setting_id)}__best_seed{seed}.npz"
        if setting_id
        else None
    )
    if exact_name:
        for path in artifacts:
            if path.name == exact_name:
                return path

    matches = []
    for path in artifacts:
        try:
            header = _read_artifact_header(path)
        except (KeyError, ValueError):
            continue
        if header["seed"] != seed:
            continue
        oa_delta = abs(header["OA"] - float(row["OA"]))
        aa_delta = abs(header["AA"] - float(row["AA"]))
        if oa_delta <= 1e-10 and aa_delta <= 1e-10:
            matches.append(path)
    return matches[0] if len(matches) == 1 else None


def _color_mapping(classes):
    # Create a shared class palette with label zero reserved for background.

    colors = [(0.0, 0.0, 0.0, 1.0)]
    colors.extend(plt.get_cmap("tab20")(np.linspace(0, 1, classes.size)))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(classes.size + 2) - 0.5, cmap.N)
    return colors, cmap, norm


def _encode_labels(values, classes):
    # Map possibly nonconsecutive class IDs to contiguous display indices.

    encoded = np.zeros(np.asarray(values).shape, dtype=int)
    for position, label in enumerate(classes, start=1):
        encoded[np.asarray(values) == label] = position
    return encoded


def _save_map(array, output_path, cmap, norm, dpi):
    # Save a single label image without axes or surrounding whitespace.

    rows, columns = array.shape
    width = 5.0
    height = max(2.0, width * rows / columns)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    axis.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_axis_off()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _save_query_map(
    gt_encoded,
    labeled_indices,
    ground_truth,
    classes,
    colors,
    cmap,
    norm,
    output_path,
    dpi,
):
    # Overlay queried pixels on a faint ground-truth map.

    rows, columns = gt_encoded.shape
    width = 5.0
    height = max(2.0, width * rows / columns)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    axis.imshow(
        gt_encoded,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=0.35,
    )
    query_rows, query_columns = np.unravel_index(labeled_indices, (rows, columns))
    query_colors = []
    for index in labeled_indices:
        positions = np.flatnonzero(classes == ground_truth[index])
        query_colors.append(colors[int(positions[0]) + 1] if positions.size else "white")
    axis.scatter(
        query_columns,
        query_rows,
        c=query_colors,
        edgecolors="white",
        linewidths=0.5,
        s=24,
    )
    axis.set_axis_off()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _save_combined_map(
    gt_encoded,
    pred_encoded,
    labeled_indices,
    ground_truth,
    classes,
    colors,
    cmap,
    norm,
    row,
    label,
    output_path,
    dpi,
):
    # Save ground truth, prediction, and query locations in one panel.

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(gt_encoded, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("Ground truth")
    axes[1].imshow(pred_encoded, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title(
        f"Best seed {int(row.get('seed', 0))}\n"
        f"OA={float(row['OA']):.4f}, AA={float(row['AA']):.4f}"
    )
    axes[2].imshow(
        gt_encoded,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=0.35,
    )
    query_rows, query_columns = np.unravel_index(
        labeled_indices,
        gt_encoded.shape,
    )
    query_colors = []
    for index in labeled_indices:
        positions = np.flatnonzero(classes == ground_truth[index])
        query_colors.append(colors[int(positions[0]) + 1] if positions.size else "white")
    axes[2].scatter(
        query_columns,
        query_rows,
        c=query_colors,
        edgecolors="white",
        linewidths=0.5,
        s=24,
    )
    axes[2].set_title(f"Queried pixels, B={int(row['budget'])}")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(label, fontsize=10)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_best_seed_maps(
    frame,
    labels,
    metadata,
    run_dir,
    source_path,
    output_dir,
    dataset,
    dataset_dir,
    artifact_dir,
    dpi,
):
    # Reconstruct maps for the best final-budget seed of each setting.

    best_rows = _select_best_rows(frame)
    if best_rows.empty:
        return [], ["Best-seed maps require a raw CSV with budget, OA, and AA."]

    # CSV metrics do not contain pixel-wise predictions, so companion artifacts
    # are required to reconstruct classification and query maps.
    artifacts = _artifact_candidates(run_dir, source_path, artifact_dir)
    if not artifacts:
        return [], [
            "No companion .npz artifacts were found. Curves were generated, "
            "but prediction/query maps cannot be reconstructed from CSV "
            "metrics alone."
        ]

    dataset = _infer_dataset(frame, metadata, dataset)
    if dataset is None:
        return [], [
            "Dataset could not be inferred. Pass --dataset to generate maps."
        ]

    _, ground_truth, spatial_shape = load_hsi_dataset(
        dataset,
        dataset_dir=dataset_dir,
        return_shape=True,
    )
    classes = np.unique(ground_truth[ground_truth != 0])
    colors, cmap, norm = _color_mapping(classes)
    gt_encoded = _encode_labels(ground_truth, classes).reshape(spatial_shape)

    map_dir = output_dir / "best_seed"
    map_dir.mkdir(parents=True, exist_ok=True)
    gt_path = map_dir / "ground_truth.png"
    _save_map(gt_encoded, gt_path, cmap, norm, dpi)
    outputs = [gt_path]
    warnings = []
    manifest = []

    for _, row in best_rows.iterrows():
        group = row["_figure_group"]
        label = labels.get(group, str(group))
        artifact_path = _match_artifact(row, artifacts)
        if artifact_path is None:
            warnings.append(
                f"No unique artifact matched group '{group}', seed "
                f"{int(row.get('seed', 0))}, OA={float(row['OA']):.6f}."
            )
            continue

        with np.load(artifact_path, allow_pickle=False) as artifact:
            predictions = np.asarray(artifact["predictions"]).reshape(-1)
            labeled_indices = np.asarray(
                artifact["labeled_indices"],
                dtype=int,
            ).reshape(-1)
        if predictions.size != ground_truth.size:
            warnings.append(
                f"Artifact {artifact_path.name} has {predictions.size} "
                f"predictions; expected {ground_truth.size} for {dataset}."
            )
            continue

        # Background participates in graph construction but is hidden in the
        # displayed map because it is excluded from OA/AA evaluation.
        displayed = predictions.copy()
        displayed[ground_truth == 0] = 0
        pred_encoded = _encode_labels(displayed, classes).reshape(spatial_shape)
        stem = _safe_name(group)
        prediction_path = map_dir / f"{stem}__best_prediction.png"
        query_path = map_dir / f"{stem}__queried_points.png"
        panel_path = map_dir / f"{stem}__best_seed_panels.png"
        _save_map(pred_encoded, prediction_path, cmap, norm, dpi)
        _save_query_map(
            gt_encoded,
            labeled_indices,
            ground_truth,
            classes,
            colors,
            cmap,
            norm,
            query_path,
            dpi,
        )
        _save_combined_map(
            gt_encoded,
            pred_encoded,
            labeled_indices,
            ground_truth,
            classes,
            colors,
            cmap,
            norm,
            row,
            label,
            panel_path,
            dpi,
        )
        outputs.extend((prediction_path, query_path, panel_path))
        manifest.append(
            {
                "group": group,
                "label": label,
                "seed": int(row.get("seed", 0)),
                "budget": int(row["budget"]),
                "OA": float(row["OA"]),
                "AA": float(row["AA"]),
                "artifact": str(artifact_path),
                "prediction_figure": str(prediction_path),
                "query_figure": str(query_path),
                "panel_figure": str(panel_path),
            }
        )

    if manifest:
        manifest_path = map_dir / "best_seed_manifest.csv"
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
        outputs.append(manifest_path)
    return outputs, warnings


def build_parser():
    # Define the generic figure-generation CLI.

    parser = argparse.ArgumentParser(
        description=(
            "Generate generic OA/AA curves and best-seed classification maps "
            "from a FALL run directory or result CSV."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Run directory, raw result CSV, or aggregated budget-curve CSV."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Destination directory. Defaults to INPUT/figures for a run "
            "directory, or <csv-stem>_figures beside a CSV."
        ),
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        help=(
            "Columns defining separate curves. Auto-detected from setting_id, "
            "name, method, or algorithm when omitted."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("salinasA", "paviaU_crop"),
        help="Dataset override used to reconstruct best-seed maps.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(WORKSPACE_ROOT / "datasets"),
        help="Directory containing the HSI .mat files.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Optional directory containing saved best-seed .npz artifacts.",
    )
    parser.add_argument(
        "--title",
        help="Optional title added to both performance plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in dots per inch (default: 300).",
    )
    parser.add_argument(
        "--no-curves",
        action="store_true",
        help="Do not generate OA/AA budget curves.",
    )
    parser.add_argument(
        "--no-best-seed",
        action="store_true",
        help="Do not generate best-seed prediction/query maps.",
    )
    return parser


def main():
    # Load one result source and generate the requested figure products.

    args = build_parser().parse_args()
    source_path, run_dir = _resolve_source(args.input)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else _default_output_dir(args.input, source_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, metadata_path = _load_metadata(run_dir, source_path)
    raw_frame = pd.read_csv(source_path)
    group_columns = _detect_group_columns(raw_frame, args.group_by)
    curves, grouped_frame, labels = build_curves(
        raw_frame,
        source_path,
        group_columns,
    )

    outputs = []
    warnings = []
    if not args.no_curves:
        outputs.extend(
            save_curves(
                curves,
                output_dir,
                args.title,
                args.dpi,
            )
        )
    if not args.no_best_seed:
        map_outputs, map_warnings = save_best_seed_maps(
            grouped_frame,
            labels,
            metadata,
            run_dir,
            source_path,
            output_dir,
            args.dataset,
            args.dataset_dir,
            args.artifact_dir,
            args.dpi,
        )
        outputs.extend(map_outputs)
        warnings.extend(map_warnings)

    # Record every resolved input and generated file so figures remain
    # traceable to a particular result table and artifact directory.
    run_info = {
        "input": str(Path(args.input).resolve()),
        "source_csv": str(source_path),
        "metadata": str(metadata_path) if metadata_path else None,
        "group_by": group_columns,
        "output_dir": str(output_dir),
        "generated_files": [str(path) for path in outputs],
        "warnings": warnings,
    }
    manifest_path = output_dir / "figure_generation.json"
    manifest_path.write_text(
        json.dumps(run_info, indent=2),
        encoding="ascii",
    )
    outputs.append(manifest_path)

    print("input CSV:", source_path)
    print("grouped by:", group_columns or ["single result"])
    print("saved:")
    for path in outputs:
        print(" ", path)
    for warning in warnings:
        print("warning:", warning)


if __name__ == "__main__":
    main()
