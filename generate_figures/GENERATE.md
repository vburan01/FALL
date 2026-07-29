# Figure generation

`generate.py` converts saved experiment results into reusable figures. It
accepts either:

- a run directory produced by `Fermat_repo/run_FALL.py`, or
- a raw/aggregated result CSV.

## Run directory

```powershell
python Fermat_repo/generate_figures/generate.py `
  Fermat_repo/runs/my_run
```

By default, figures are written to:

```text
Fermat_repo/runs/my_run/figures/
```

The generated files are:

```text
budget_curves.csv
oa_curve.png
aa_curve.png
figure_generation.json
best_seed/
    ground_truth.png
    <setting>__best_prediction.png
    <setting>__queried_points.png
    <setting>__best_seed_panels.png
    best_seed_manifest.csv
```

Best-seed maps are generated when the run directory contains the companion
`visualizations/*.npz` files saved by `run_FALL.py`. A metrics CSV by itself
does not contain predictions or queried indices, so it can always reproduce
the curves but cannot reproduce the maps unless `--artifact-dir` points to
those `.npz` files.

## CSV input

```powershell
python Fermat_repo/generate_figures/generate.py `
  Fermat_repo/runs/example/raw_results.csv `
  --dataset salinasA
```

For a CSV, the default output directory is
`<csv-name>_figures` beside the input file.

## Useful options

```powershell
# Choose where figures are written.
python Fermat_repo/generate_figures/generate.py RUN_DIR `
  --output-dir runs/paper_figures

# Explicitly define which columns distinguish result curves.
python Fermat_repo/generate_figures/generate.py RESULTS.csv `
  --group-by method p0

# Curves only.
python Fermat_repo/generate_figures/generate.py RESULTS.csv `
  --no-best-seed

# Maps only.
python Fermat_repo/generate_figures/generate.py RUN_DIR `
  --no-curves
```

Raw CSVs must contain `budget`, `OA`, and `AA`; `seed` is used when present.
Aggregated CSVs must contain `budget`, `OA_mean`, and `AA_mean`.
