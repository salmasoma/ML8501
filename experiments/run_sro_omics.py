"""Fixed end-to-end experiment runner for Iterative SRO on omics data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        "--count-size",
        type=int,
        default=None,
        help="Number of CountSketch rows when using count/count_gaussian (default: 2x sketch-size or n_samples).",
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
        "--figure-dir",
        type=Path,
        default=None,
        help="Optional directory where convergence and comparison plots will be saved.",
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
    if n_train <= 0:
        return [("none", SketchConfig(method="none"))]

    sketch_size = max(0, min(args.sketch_size, n_train))
    count_size = (
        min(args.count_size, n_train)
        if args.count_size is not None
        else min(n_train, max(sketch_size, 1) * 2)
    )

    configs: List[Tuple[str, SketchConfig]] = [
        ("none", SketchConfig(method="none")),
    ]

    if sketch_size > 0:
        configs.append(
            (
                "gaussian",
                SketchConfig(
                    method="gaussian",
                    sketch_size=sketch_size,
                    random_state=args.random_state,
                ),
            )
        )
        configs.append(
            (
                "srht",
                SketchConfig(
                    method="srht",
                    sketch_size=sketch_size,
                    random_state=args.random_state,
                ),
            )
        )

    if count_size and count_size > 0:
        configs.append(
            (
                "count",
                SketchConfig(
                    method="count",
                    sketch_size=count_size,
                    random_state=args.random_state,
                ),
            )
        )
        if sketch_size > 0:
            configs.append(
                (
                    "count_gaussian",
                    SketchConfig(
                        method="count_gaussian",
                        sketch_size=sketch_size,
                        count_size=count_size,
                        random_state=args.random_state,
                    ),
                )
            )

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
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Center the target variable
    y_mean = np.mean(y_train)
    y_train_centered = y_train - y_mean
    
    results: List[Dict[str, object]] = []
    histories: List[Dict[str, object]] = []

    # Baseline models
    baselines = {
        "Ridge": Ridge(alpha=args.ridge_strength),
        "Lasso": Lasso(
            alpha=args.lasso_strength,
            max_iter=10000,
            tol=1e-4,
            random_state=args.random_state,
        ),
    }

    for name, model in baselines.items():
        try:
            model.fit(X_train_scaled, y_train_centered)
            y_pred = model.predict(X_test_scaled) + y_mean  # Add back the mean
            results.append(
                _make_result_row(
                    name,
                    "baseline",
                    y_test,
                    y_pred,
                    regularizer=name,
                    subspace="baseline",
                )
            )
        except Exception as e:
            print(f"Warning: {name} baseline failed: {e}")
            # Add a dummy result to show the failure
            results.append(
                _make_result_row(
                    name + " (failed)",
                    "baseline",
                    y_test,
                    np.full_like(y_test, y_mean),
                    regularizer=name,
                    subspace="baseline",
                )
            )

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
                results.append(
                    _make_result_row(
                        tag,
                        "sro",
                        y_test,
                        y_pred,
                        regularizer=reg_name,
                        subspace=sketch_name,
                    )
                )
                history_frame = pd.DataFrame(solver.get_history())
                if not history_frame.empty:
                    histories.append(
                        {
                            "model": tag,
                            "regularizer": reg_name,
                            "subspace": sketch_name,
                            "history": history_frame,
                        }
                    )
                _maybe_dump_history(args.history_dir, tag, solver)
            except Exception as e:
                print(f"Warning: {tag} failed: {e}")
                # Add a dummy result to show the failure
                results.append(
                    _make_result_row(
                        tag + " (failed)",
                        "sro",
                        y_test,
                        np.full_like(y_test, y_mean),
                        regularizer=reg_name,
                        subspace=sketch_name,
                    )
                )

    return pd.DataFrame(results), histories


def _make_result_row(
    model_name: str,
    family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    regularizer: Optional[str] = None,
    subspace: Optional[str] = None,
) -> Dict[str, object]:
    # Ensure predictions are finite
    y_pred = np.where(np.isfinite(y_pred), y_pred, np.mean(y_true))
    
    row = {
        "model": model_name,
        "family": family,
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }
    if regularizer is not None:
        row["regularizer"] = regularizer
    if subspace is not None:
        row["subspace"] = subspace
    return row


def _maybe_dump_history(history_dir: Path | None, tag: str, solver: IterativeSRO) -> None:
    if history_dir is None:
        return
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{tag}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(solver.get_history(), file, indent=2)


def create_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mae", "mse", "rmse", "r2"]
    if not set(["regularizer", "subspace"]).issubset(results.columns):
        return (
            results.set_index(["family", "model"])[metrics]
            if "model" in results.columns
            else results[metrics]
        )

    enriched = results.copy()
    enriched["regularizer"] = enriched["regularizer"].fillna(enriched["model"])
    enriched["subspace"] = enriched["subspace"].fillna("baseline")

    pivot = enriched.pivot_table(
        index=["family", "regularizer"],
        columns="subspace",
        values=metrics,
    ).sort_index()

    if isinstance(pivot.columns, pd.MultiIndex):
        pivot = pivot.sort_index(axis=1, level=0)
        pivot.columns = [f"{subspace}_{metric}" for metric, subspace in pivot.columns]

    return pivot


def generate_visualisations(
    results: pd.DataFrame,
    histories: List[Dict[str, object]],
    figure_dir: Path | None,
) -> None:
    if figure_dir is None:
        return

    sns.set_theme(style="whitegrid")
    plot_metric_bars(results, figure_dir)
    plot_convergence(histories, figure_dir)


def plot_metric_bars(results: pd.DataFrame, figure_dir: Path) -> None:
    metrics = ["mae", "rmse", "r2"]
    plot_data = results.copy()
    plot_data["regularizer"] = plot_data["regularizer"].fillna(plot_data["model"])
    plot_data["subspace"] = plot_data["subspace"].fillna("baseline")

    order_map = {
        "baseline": 0,
        "none": 1,
        "gaussian": 2,
        "count": 3,
        "count_gaussian": 4,
        "srht": 5,
    }
    plot_data["subspace_order"] = plot_data["subspace"].map(order_map).fillna(99)
    subspace_order = (
        plot_data.sort_values("subspace_order")["subspace"].drop_duplicates().tolist()
    )

    regularizer_labels = plot_data["regularizer"].astype(str)
    sro_mask = plot_data["family"] == "sro"
    regularizer_labels.loc[sro_mask] = (
        regularizer_labels.loc[sro_mask].str.replace("_", " ").str.title()
    )
    plot_data["regularizer_label"] = regularizer_labels
    plot_data["reg_order"] = np.where(plot_data["family"] == "baseline", 0, 1)
    regularizer_order = (
        plot_data.sort_values(["reg_order", "regularizer_label"])["regularizer_label"]
        .drop_duplicates()
        .tolist()
    )

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=plot_data,
            x="subspace",
            y=metric,
            hue="regularizer_label",
            order=subspace_order,
            hue_order=regularizer_order,
            ax=ax,
        )
        ax.set_xlabel("Subspace / Sketch Type")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} comparison across subspaces")
        ax.legend(title="Regulariser")
        fig.tight_layout()
        output_path = figure_dir / f"{metric}_comparison.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def plot_convergence(histories: List[Dict[str, object]], figure_dir: Path) -> None:
    if not histories:
        return

    def _label(entry: Dict[str, object]) -> str:
        regularizer = str(entry.get("regularizer", ""))
        regularizer = regularizer.replace("_", " ").title()
        subspace = str(entry.get("subspace", ""))
        return f"{regularizer} ({subspace})"

    objective_fig, objective_ax = plt.subplots(figsize=(10, 6))
    beta_fig, beta_ax = plt.subplots(figsize=(10, 6))

    for entry in histories:
        history_frame = entry.get("history")
        if history_frame is None or history_frame.empty:
            continue
        label = _label(entry)
        if "iteration" not in history_frame.columns:
            history_frame = history_frame.reset_index().rename(columns={"index": "iteration"})

        if "objective" in history_frame.columns:
            objective_ax.plot(history_frame["iteration"], history_frame["objective"], label=label)
        if "beta_change" in history_frame.columns:
            beta_ax.plot(history_frame["iteration"], history_frame["beta_change"], label=label)

    if objective_ax.lines:
        objective_ax.set_xlabel("Iteration")
        objective_ax.set_ylabel("Objective value")
        objective_ax.set_title("Convergence of objective values")
        objective_ax.legend()
        objective_ax.grid(True)
        objective_fig.tight_layout()
        objective_fig.savefig(figure_dir / "convergence_objective.png", dpi=300)
    plt.close(objective_fig)

    if beta_ax.lines:
        beta_ax.set_xlabel("Iteration")
        beta_ax.set_ylabel(r"$\\|\\beta^{(t)} - \\beta^{(t-1)}\\|_2$")
        beta_ax.set_title("Iterate change across iterations")
        beta_ax.legend()
        beta_ax.grid(True)
        beta_fig.tight_layout()
        beta_fig.savefig(figure_dir / "convergence_beta_change.png", dpi=300)
    plt.close(beta_fig)


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args)
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"Target variable stats: mean={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    results, histories = evaluate_models(X_train, X_test, y_train, y_test, args)
    results = results.sort_values(by="mse").reset_index(drop=True)

    print("\nModel comparison (sorted by MSE):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))

    if args.figure_dir is not None:
        args.figure_dir.mkdir(parents=True, exist_ok=True)

    comparison_table = create_comparison_table(results)
    print("\nDetailed comparison by family, regulariser and subspace:")
    print(comparison_table.round(4).to_string())

    if args.figure_dir is not None:
        table_path = args.figure_dir / "comparison_table.csv"
        comparison_table.to_csv(table_path)
        print(f"Comparison table saved to {table_path}")

    generate_visualisations(results, histories, args.figure_dir)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
