# Iterative SRO for MMSE prediction

This repository provides a reference implementation of the iterative
successive regularized optimisation (SRO) algorithm from
[Sketching for convex and nonconvex regularized least squares with sharp guarantees](https://openreview.net/pdf?id=7liN6uHAQZ)
and applies it to MMSE prediction tasks on omics datasets.

The code supports both convex (ridge, lasso) and non-convex (SCAD)
regularisers and offers multiple sketching strategies: Gaussian,
CountSketch, CountSketch+Gaussian hybrids and SRHT embeddings implemented
through [`pylspack`](https://github.com/IBM/pylspack).

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

The main experiment driver lives in `experiments/run_sro_omics.py`. It can work
with a CSV omics dataset or generate a synthetic benchmark.

### Synthetic quick start

```bash
python experiments/run_sro_omics.py --synthetic --iterations 4 --inner-iterations 60 \
    --sketch-size 64 --count-size 128 --output results.csv
```

### Real omics data

```bash
python experiments/run_sro_omics.py --data path/to/omics.csv --target mmse_column \
    --drop-columns patient_id --iterations 6 --inner-iterations 120 --history-dir histories
```

Command-line arguments let you control the choice of regulariser,
sketch sizes (including dedicated CountSketch row counts), stopping
tolerance and whether sketches are resampled per iteration. The script
prints a comparison table across baselines (ridge, lasso) and every SRO
configuration. Visual summaries are written to `./figures` by default; use
`--figure-dir` to customise the output location or `--no-figures` to skip
plotting entirely. The grouped bar charts cover MAE/RMSE/R² across
subspaces and convergence plots are built from the stored optimisation
histories. Optionally, optimisation histories are saved as JSON files via
`--history-dir`.

## Module overview

- `sro/regularizers.py` implements the proximal operators and penalties for
  L2, L1 and SCAD regularisers.
- `sro/sketching.py` wraps the sketch primitives, delegating CountSketch
  and hybrid transforms to `pylspack`.
- `sro/sro_solver.py` contains the iterative SRO optimizer capable of
  handling convex and non-convex penalties.
- `experiments/run_sro_omics.py` orchestrates data loading, model fitting
  and metric reporting on omics→MMSE tasks.

## Reproducibility

Set the `--random-state` flag to enforce deterministic data splits and
Gaussian sketch matrices. CountSketch transforms rely on `pylspack`'s
internal RNG, so runs may differ slightly across executions when sketch
resampling is enabled.
