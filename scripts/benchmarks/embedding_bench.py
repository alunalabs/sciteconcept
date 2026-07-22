#!/usr/bin/env python3
"""Run and summarize the frozen-embedding benchmark.

Two executable surfaces are provided:

``mechanism`` scores masked sender/receiver role embeddings with the exact
group-held-out pair probe used by the mechanism pillar. It consumes a records
CSV and one or more already encoded ``[records, dimensions]`` NumPy arrays.
The encoder is therefore replaceable; use ``scripts/encode_cells.py`` to create
sCITEconcept embeddings.

``summarize`` converts the 20 normalized component rows per model into pillar,
representation, and deployment scores. This is the path exercised by the
public notebook against the checked-in component table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from ._common import load_embeddings, parse_assignment, unit_rows
except ImportError:  # direct execution
    from _common import load_embeddings, parse_assignment, unit_rows


METRIC_WEIGHTS = {
    "state_transfer": {
        "leave_dataset_cell_type_macro_f1": 0.35,
        "leave_platform_cell_type_macro_f1": 0.35,
        "masked_inhibitory_program_spearman_unit": 0.20,
        "cell_type_tissue_map_at50": 0.10,
    },
    "mechanism_pairing": {
        "heldout_interaction_pair_auprc_lift_scaled": 0.35,
        "heldout_tissue_pair_auprc_lift_scaled": 0.25,
        "inhibitory_pair_macro_f1": 0.25,
        "coexpressive_pair_macro_f1": 0.15,
    },
    "perturbation_observability": {
        "mapped_target_ko_retrieval_map": 0.30,
        "cross_context_delta_direction_agreement_unit": 0.25,
        "ko_vs_reencode_noise_auc": 0.20,
        "post_ko_identity_retention_at10": 0.15,
        "target_masked_response_program_spearman_unit": 0.10,
    },
    "native_panel_invariance": {
        "paired_view_mrr": 0.35,
        "paired_view_recall_at50": 0.30,
        "cross_panel_cell_type_macro_f1": 0.20,
        "full_to_panel_biology_retention": 0.15,
    },
    "coverage": {
        "target_mappable_fraction": 0.50,
        "nonzero_intervention_fraction": 0.30,
        "finite_embedding_fraction": 0.20,
    },
}
PILLAR_WEIGHTS = {
    "state_transfer": 0.30,
    "mechanism_pairing": 0.25,
    "perturbation_observability": 0.25,
    "native_panel_invariance": 0.15,
    "coverage": 0.05,
}
BIOLOGICAL_PILLARS = tuple(name for name in PILLAR_WEIGHTS if name != "coverage")
GEOMETRIC_FLOOR = 0.01


def weighted_geometric_mean(values: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Geometric-mean weights must be positive")
    return math.exp(
        sum(
            weight * math.log(max(GEOMETRIC_FLOOR, float(values[name])))
            for name, weight in weights.items()
        )
        / total
    )


def summarize_metric_rows(metric_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct all pillar and composite scores from normalized metric rows."""

    required = {"model", "pillar", "metric", "estimate", "n_units"}
    missing = required - set(metric_rows.columns)
    if missing:
        raise ValueError(f"Metric table is missing columns: {sorted(missing)}")
    frame = metric_rows.copy()
    frame["estimate"] = pd.to_numeric(frame["estimate"], errors="raise")
    if not frame["estimate"].between(0.0, 1.0).all():
        raise ValueError("Every normalized component estimate must be in [0, 1]")
    if frame.duplicated(["model", "pillar", "metric"]).any():
        raise ValueError("Duplicate model/pillar/metric rows")

    pillar_rows: list[dict[str, Any]] = []
    for model, model_rows in frame.groupby("model", sort=False):
        for pillar, weights in METRIC_WEIGHTS.items():
            current = model_rows[model_rows["pillar"].eq(pillar)].set_index("metric")
            if set(current.index) != set(weights):
                raise ValueError(
                    f"{model}/{pillar} metrics differ from contract: "
                    f"{sorted(set(current.index) ^ set(weights))}"
                )
            score = sum(float(current.loc[name, "estimate"]) * weight for name, weight in weights.items())
            pillar_rows.append(
                {
                    "model": model,
                    "pillar": pillar,
                    "pillar_score": score,
                    "pillar_weight": PILLAR_WEIGHTS[pillar],
                }
            )
    pillars = pd.DataFrame(pillar_rows)

    leaderboard_rows = []
    for model, model_rows in pillars.groupby("model", sort=False):
        values = model_rows.set_index("pillar")["pillar_score"].to_dict()
        if set(values) != set(PILLAR_WEIGHTS):
            raise ValueError(f"Incomplete pillar inventory for {model}")
        representation_weights = {
            name: PILLAR_WEIGHTS[name] for name in BIOLOGICAL_PILLARS
        }
        leaderboard_rows.append(
            {
                "model": model,
                "representation_score": weighted_geometric_mean(values, representation_weights),
                "deployment_score": weighted_geometric_mean(values, PILLAR_WEIGHTS),
                "representation_eligible": True,
                "deployment_eligible": True,
                "deployment_failed_gates": "",
            }
        )
    return pillars, pd.DataFrame(leaderboard_rows)


def pair_feature(sender: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    sender_unit = unit_rows(np.asarray(sender).reshape(1, -1))[0]
    receiver_unit = unit_rows(np.asarray(receiver).reshape(1, -1))[0]
    return np.concatenate(
        [sender_unit, receiver_unit, np.abs(sender_unit - receiver_unit), sender_unit * receiver_unit]
    ).astype(np.float32)


def build_mechanism_examples(records: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Build one true and three matched-negative pairs per signaling context."""

    required = {"context_key", "role", "interaction", "relationship", "tissue"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Mechanism records are missing columns: {sorted(missing)}")
    rows = []
    subset = records[records.get("task", "mechanism").eq("mechanism")] if "task" in records else records
    for context_key, context in subset.groupby("context_key", sort=True):
        means: dict[str, np.ndarray] = {}
        for role in ("sender_pos", "receiver_pos", "sender_neg", "receiver_neg"):
            indices = context.index[context["role"].eq(role)].to_numpy(np.int64)
            if indices.size == 0:
                break
            means[role] = unit_rows(embeddings[indices]).mean(axis=0)
        if len(means) != 4:
            continue
        metadata = context.iloc[0]
        for pair_kind, sender_role, receiver_role, target in (
            ("positive", "sender_pos", "receiver_pos", 1),
            ("receiver_negative", "sender_pos", "receiver_neg", 0),
            ("sender_negative", "sender_neg", "receiver_pos", 0),
            ("double_negative", "sender_neg", "receiver_neg", 0),
        ):
            rows.append(
                {
                    "context_key": context_key,
                    "interaction": metadata["interaction"],
                    "relationship": metadata["relationship"],
                    "tissue": metadata["tissue"],
                    "pair_kind": pair_kind,
                    "target": target,
                    "feature": pair_feature(means[sender_role], means[receiver_role]),
                }
            )
    if not rows:
        raise ValueError("No complete mechanism contexts were found")
    return pd.DataFrame(rows)


def grouped_pair_probe(
    examples: pd.DataFrame, group_column: str, *, seed: int = 20260709
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit PCA plus a balanced ridge probe with whole groups held out."""

    x = np.stack(examples["feature"].to_numpy())
    y = examples["target"].to_numpy(np.int64)
    groups = examples[group_column].astype(str).to_numpy()
    n_groups = int(np.unique(groups).size)
    if n_groups < 2:
        raise ValueError(f"Need at least two {group_column} groups")
    splitter = GroupKFold(n_splits=min(5, n_groups))
    fold_rows = []
    prediction_rows = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
        components = min(256, len(train) - 1, x.shape[1])
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, whiten=True, svd_solver="randomized", random_state=seed),
            RidgeClassifier(alpha=1.0, class_weight="balanced"),
        )
        model.fit(x[train], y[train])
        decision = np.asarray(model.decision_function(x[test]), dtype=np.float64)
        predicted = np.asarray(model.predict(x[test]), dtype=np.int64)
        prevalence = float(np.mean(y[test]))
        average_precision = float(average_precision_score(y[test], decision))
        lift = float(
            np.clip(
                (average_precision - prevalence) / max(1.0 - prevalence, 1.0e-12),
                0.0,
                1.0,
            )
        )
        fold_rows.append(
            {
                "fold": fold,
                "group_column": group_column,
                "n_test": int(len(test)),
                "prevalence": prevalence,
                "average_precision": average_precision,
                "auprc_lift_scaled": lift,
                "macro_f1": float(f1_score(y[test], predicted, average="macro", zero_division=0)),
            }
        )
        for local, row_index in enumerate(test.tolist()):
            prediction_rows.append(
                {
                    "row_index": row_index,
                    "fold": fold,
                    "truth": int(y[row_index]),
                    "prediction": int(predicted[local]),
                    "decision": float(decision[local]),
                    "relationship": str(examples.iloc[row_index]["relationship"]),
                    "context_key": str(examples.iloc[row_index]["context_key"]),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def score_mechanism(records: pd.DataFrame, embeddings: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    examples = build_mechanism_examples(records, embeddings)
    interaction_folds, predictions = grouped_pair_probe(examples, "interaction")
    tissue_folds, _ = grouped_pair_probe(examples, "tissue")
    metrics = {
        "heldout_interaction_pair_auprc_lift_scaled": float(interaction_folds["auprc_lift_scaled"].mean()),
        "heldout_tissue_pair_auprc_lift_scaled": float(tissue_folds["auprc_lift_scaled"].mean()),
    }
    for relationship, name in (
        ("inhibitory", "inhibitory_pair_macro_f1"),
        ("coexpressive", "coexpressive_pair_macro_f1"),
    ):
        current = predictions[predictions["relationship"].eq(relationship)]
        metrics[name] = float(
            f1_score(current["truth"], current["prediction"], average="macro", zero_division=0)
        )
    details = pd.concat(
        [
            interaction_folds.assign(surface="leave_interaction"),
            tissue_folds.assign(surface="leave_tissue"),
        ],
        ignore_index=True,
    )
    return metrics, details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize", help="Build pillars and composites")
    summarize.add_argument("--metric-rows", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    mechanism = subparsers.add_parser("mechanism", help="Score role-record embeddings")
    mechanism.add_argument("--records", type=Path, required=True)
    mechanism.add_argument("--embedding", action="append", required=True, metavar="NAME=PATH")
    mechanism.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "summarize":
        pillars, leaderboard = summarize_metric_rows(pd.read_csv(args.metric_rows))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pillars.to_csv(args.output_dir / "embedding_bench_pillar_scores.csv", index=False)
        leaderboard.to_csv(args.output_dir / "embedding_bench_leaderboard.csv", index=False)
        print(leaderboard.to_string(index=False))
        return

    records = pd.read_csv(args.records).reset_index(drop=True)
    rows = []
    fold_tables = []
    for assignment in args.embedding:
        model_name, path = parse_assignment(assignment)
        metrics, folds = score_mechanism(
            records, load_embeddings(path, expected_rows=len(records))
        )
        rows.extend(
            {
                "model": model_name,
                "pillar": "mechanism_pairing",
                "metric": metric,
                "estimate": estimate,
                "n_units": int(records["context_key"].nunique()),
            }
            for metric, estimate in metrics.items()
        )
        fold_tables.append(folds.assign(model=model_name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(
        args.output.with_name(args.output.stem + "_folds.csv"), index=False
    )


if __name__ == "__main__":
    main()
