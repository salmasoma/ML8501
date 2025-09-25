"""Fixed end-to-end experiment runner for Iterative SRO on omics data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sro import (
    L1Regularizer,
    L2Regularizer,
    SCADRegularizer,
    SketchConfig,
    IterativeSRO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to CSV file containing the dataset.")
    parser.add_argument(
        "--target",
        default="mmse",
        help="Name of the MMSE target column in the dataset (default: mmse).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=None,
        help="Optional list of columns to drop before modelling.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for the test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=13,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of outer SRO iterations (default: 5).",
    )
    parser.add_argument(
        "--inner-iterations",
        type=int,
        default=100,
        help="Number of proximal updates per SRO iteration (default: 100).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Stopping tolerance for both inner and outer loops.",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=0.5,  # Reduced from 1.0 for better stability
        help="Step size multiplier relative to the Lipschitz constant (default: 0.5).",
    )
    parser.add_argument(
        "--sketch-size",
        type=int,
        default=128,
        help="Number of rows for sketches (default: 128).",
    )
    parser.add_argument(
        "--lasso-strength",
        type=float,
        default=0.01,  # Reduced from 0.05
        help="Lambda parameter for the L1 regulariser (default: 0.01).",
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=1.0,
        help="Lambda parameter for the L2 regulariser (default: 1.0).",
    )
    parser.add_argument(
        "--scad-strength",
        type=float,
        default=0.01,  # Reduced from 0.05
        help="Lambda parameter for the SCAD regulariser (default: 0.01).",
    )
    parser.add_argument(
        "--scad-a",
        type=float,
        default=3.7,
        help="SCAD shape parameter a (default: 3.7).",
    )
    parser.add_argument(
        "--regularizers",
        nargs="*",
        default=("lasso", "scad"),
        choices=("lasso", "ridge", "scad"),
        help="Regularisers to evaluate with SRO (default: lasso scad).",
    )
    parser.add_argument(
        "--fixed-sketch",
        action="store_true",
        help="Reuse the same sketch at every SRO iteration instead of resampling.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="Optional directory where per-run optimisation traces are stored as JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the metrics table as CSV.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic omics-like dataset instead of loading from disk.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of synthetic samples to generate when --synthetic is used.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=200,
        help="Number of synthetic omics features when --synthetic is used.",
    )
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.synthetic:
        X, y = _generate_synthetic(args.n_samples, args.n_features, args.random_state)
        return X, y

    if args.data is None:
        raise ValueError("--data must be provided unless --synthetic is set.")
    frame = pd.read_csv(args.data)
    if args.drop_columns:
        frame = frame.drop(columns=list(args.drop_columns), errors="ignore")

    if args.target not in frame.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")

    frame = frame.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    y = frame.pop(args.target).to_numpy(dtype=float)
    X = frame.to_numpy(dtype=float)
    return X, y


def _generate_synthetic(n_samples: int, n_features: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    ground_truth = rng.normal(scale=0.3, size=n_features)
    y = X @ ground_truth + rng.normal(scale=0.5, size=n_samples)
    y = np.clip(y, 0, 30)  # mimic MMSE bounds
    return X, y


def build_sro_configs(
    args: argparse.Namespace, n_train: int
) -> Iterable[Tuple[str, SketchConfig]]:
    sketch_size = min(args.sketch_size, n_train)
    configs = [
        ("none", SketchConfig(method="none")),
    ]
    # Only add sketching if we have a reasonable number of samples
    if sketch_size >= 32 and sketch_size < n_train:
        configs.append(("subsampled", SketchConfig(method="subsampling", sketch_size=sketch_size, random_state=args.random_state)))
    return configs


def build_regularizers(args: argparse.Namespace) -> Dict[str, object]:
    registry = {
        "lasso": L1Regularizer(args.lasso_strength),
        "ridge": L2Regularizer(args.ridge_strength),
        "scad": SCADRegularizer(args.scad_strength, a=args.scad_a),
    }
    return {name: registry[name] for name in args.regularizers}


def evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Center the target variable
    y_mean = np.mean(y_train)
    y_train_centered = y_train - y_mean
    
    results: List[Dict[str, object]] = []

    # Baseline models
    baselines = {
        "Ridge": Ridge(alpha=args.ridge_strength, random_state=args.random_state),
        "Lasso": Lasso(alpha=args.lasso_strength, max_iter=10000, tol=1e-4, random_state=args.random_state),
    }

    for name, model in baselines.items():
        try:
            model.fit(X_train_scaled, y_train_centered)
            y_pred = model.predict(X_test_scaled) + y_mean  # Add back the mean
            results.append(_make_result_row(name, "baseline", y_test, y_pred))
        except Exception as e:
            print(f"Warning: {name} baseline failed: {e}")
            # Add a dummy result to show the failure
            results.append(_make_result_row(name + " (failed)", "baseline", y_test, np.full_like(y_test, y_mean)))

    # SRO models
    regularizers = build_regularizers(args)
    sketch_configs = list(build_sro_configs(args, X_train.shape[0]))

    for reg_name, regularizer in regularizers.items():
        for sketch_name, sketch_config in sketch_configs:
            tag = f"SRO-{reg_name}-{sketch_name}"
            try:
                solver = IterativeSRO(
                    regularizer=regularizer,
                    sketch_config=sketch_config,
                    max_iter=args.iterations,
                    inner_max_iter=args.inner_iterations,
                    tol=args.tol,
                    step_scale=args.step_scale,
                    resample_sketch=not args.fixed_sketch,
                    random_state=args.random_state,
                )
                solver.fit(X_train_scaled, y_train_centered)
                y_pred = solver.predict(X_test_scaled) + y_mean  # Add back the mean
                results.append(_make_result_row(tag, "sro", y_test, y_pred))
                _maybe_dump_history(args.history_dir, tag, solver)
            except Exception as e:
                print(f"Warning: {tag} failed: {e}")
                # Add a dummy result to show the failure
                results.append(_make_result_row(tag + " (failed)", "sro", y_test, np.full_like(y_test, y_mean)))

    return pd.DataFrame(results)


def _make_result_row(model_name: str, family: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    # Ensure predictions are finite
    y_pred = np.where(np.isfinite(y_pred), y_pred, np.mean(y_true))
    
    return {
        "model": model_name,
        "family": family,
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def _maybe_dump_history(history_dir: Path | None, tag: str, solver: IterativeSRO) -> None:
    if history_dir is None:
        return
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{tag}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(solver.get_history(), file, indent=2)


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args)
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"Target variable stats: mean={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    results = evaluate_models(X_train, X_test, y_train, y_test, args)
    results = results.sort_values(by="mse").reset_index(drop=True)

    print("\nModel comparison (sorted by MSE):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
