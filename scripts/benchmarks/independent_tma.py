#!/usr/bin/env python3
"""Run the independent serial-section spatial-platform benchmark.

``score`` consumes one prototype embedding matrix per model plus the public
prototype metadata contract. It evaluates matched-core retrieval,
cross-platform cell-type projection, and leave-patient-out tumor/PD-L1 state
transfer. ``summarize`` reconstructs the released headline tables from the
included directed-view-pair results.

The encoder is deliberately outside this module: embeddings may be produced by
``scripts/encode_cells.py`` or any comparator that preserves prototype order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

try:
    from ._common import load_embeddings, parse_assignment, unit_rows
except ImportError:  # direct execution: python scripts/benchmarks/independent_tma.py
    from _common import load_embeddings, parse_assignment, unit_rows


RETRIEVAL_METRICS = ("pairwise_auc", "mrr", "recall_at_1", "recall_at_5")


def residualize(values: np.ndarray, frame: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Subtract each assay-context mean, then L2-normalize every prototype."""

    output = np.asarray(values, dtype=np.float64).copy()
    for indices in frame.groupby(keys, sort=False, observed=True).indices.values():
        output[indices] -= output[indices].mean(axis=0, keepdims=True)
    return unit_rows(output)


def ordered_cross_platform_view_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    mapping_frame = frame[["view", "platform"]].drop_duplicates()
    if mapping_frame["view"].duplicated().any():
        raise ValueError("Each assay view must map to exactly one platform")
    mapping = dict(zip(mapping_frame["view"], mapping_frame["platform"], strict=True))
    views = sorted(mapping)
    return [
        (source, target)
        for source in views
        for target in views
        if source != target and mapping[source] != mapping[target]
    ]


def core_retrieval_queries(
    *,
    model: str,
    values: np.ndarray,
    frame: pd.DataFrame,
    min_candidates: int = 4,
) -> pd.DataFrame:
    """Rank the true serial-section core among biology-matched hard negatives."""

    contexts = ["source_tma", "source_tissue", "tumor", "cell_type"]
    rows: list[dict[str, Any]] = []
    for source_view, target_view in ordered_cross_platform_view_pairs(frame):
        source_groups = frame[frame["view"].eq(source_view)].groupby(
            contexts, sort=False, observed=True
        ).groups
        target_groups = frame[frame["view"].eq(target_view)].groupby(
            contexts, sort=False, observed=True
        ).groups
        for context in source_groups.keys() & target_groups.keys():
            source_index = np.asarray(source_groups[context], dtype=np.int64)
            target_index = np.asarray(target_groups[context], dtype=np.int64)
            if target_index.size < min_candidates:
                continue
            target_cores = frame.iloc[target_index]["source_core_key"].astype(str).to_numpy()
            if np.unique(target_cores).size != target_cores.size:
                raise ValueError(f"Duplicate target core in {target_view}/{context}")
            target_lookup = {core: index for index, core in enumerate(target_cores)}
            target_values = values[target_index]
            for source_row in source_index:
                metadata = frame.iloc[int(source_row)]
                true_position = target_lookup.get(str(metadata["source_core_key"]))
                if true_position is None:
                    continue
                similarities = values[int(source_row)] @ target_values.T
                true_similarity = float(similarities[true_position])
                negatives = np.delete(similarities, true_position)
                rank = int(1 + np.sum(similarities > true_similarity))
                auc = float(
                    (
                        np.sum(true_similarity > negatives)
                        + 0.5 * np.sum(true_similarity == negatives)
                    )
                    / max(1, negatives.size)
                )
                rows.append(
                    {
                        "model": model,
                        "source_view": source_view,
                        "target_view": target_view,
                        "source_tma": str(metadata["source_tma"]),
                        "source_prototype_id": int(metadata["prototype_id"]),
                        "source_core_key": str(metadata["source_core_key"]),
                        "candidate_count": int(target_index.size),
                        "rank": rank,
                        "reciprocal_rank": 1.0 / rank,
                        "recall_at_1": float(rank <= 1),
                        "recall_at_5": float(rank <= 5),
                        "pairwise_auc": auc,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_core_retrieval(
    queries: pd.DataFrame, *, min_pair_queries: int = 20
) -> pd.DataFrame:
    required = {
        "model", "source_tma", "source_view", "target_view", "rank",
        "candidate_count", "reciprocal_rank", "pairwise_auc", "recall_at_1",
        "recall_at_5",
    }
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"Core query table is missing columns: {sorted(missing)}")
    pair = (
        queries.groupby(
            ["model", "source_tma", "source_view", "target_view"],
            as_index=False,
            observed=True,
        )
        .agg(
            queries=("rank", "size"),
            candidate_count_mean=("candidate_count", "mean"),
            mrr=("reciprocal_rank", "mean"),
            recall_at_1=("recall_at_1", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            pairwise_auc=("pairwise_auc", "mean"),
        )
    )
    return pair[pair["queries"].ge(min_pair_queries)].reset_index(drop=True)


def cell_type_projection_queries(
    *, model: str, values: np.ndarray, frame: pd.DataFrame
) -> pd.DataFrame:
    """Project cell type across platforms with leave-patient-out centroids."""

    contexts = ["source_tma", "source_tissue", "tumor"]
    rows: list[dict[str, Any]] = []
    for source_view, target_view in ordered_cross_platform_view_pairs(frame):
        source_groups = frame[frame["view"].eq(source_view)].groupby(
            contexts, sort=False, observed=True
        ).groups
        target_groups = frame[frame["view"].eq(target_view)].groupby(
            contexts, sort=False, observed=True
        ).groups
        for context in source_groups.keys() & target_groups.keys():
            source_index = np.asarray(source_groups[context], dtype=np.int64)
            target_index = np.asarray(target_groups[context], dtype=np.int64)
            labels = frame.iloc[source_index]["cell_type"].astype(str).to_numpy()
            patients = frame.iloc[source_index]["source_patient_key"].astype(str).to_numpy()
            for target_row in target_index:
                metadata = frame.iloc[int(target_row)]
                candidate_labels: list[str] = []
                centroids: list[np.ndarray] = []
                for label in sorted(np.unique(labels)):
                    keep = (labels == label) & (
                        patients != str(metadata["source_patient_key"])
                    )
                    if np.sum(keep) < 2 or np.unique(patients[keep]).size < 2:
                        continue
                    candidate_labels.append(str(label))
                    centroids.append(
                        unit_rows(values[source_index[keep]].mean(axis=0, keepdims=True))[0]
                    )
                if str(metadata["cell_type"]) not in candidate_labels or len(centroids) < 2:
                    continue
                scores = values[int(target_row)] @ np.asarray(centroids).T
                rows.append(
                    {
                        "model": model,
                        "source_tma": str(metadata["source_tma"]),
                        "source_view": source_view,
                        "target_view": target_view,
                        "true_label": str(metadata["cell_type"]),
                        "predicted_label": candidate_labels[int(np.argmax(scores))],
                    }
                )
    return pd.DataFrame(rows)


def aggregate_classification(
    queries: pd.DataFrame, *, min_pair_queries: int = 20
) -> pd.DataFrame:
    required = {
        "model", "source_tma", "source_view", "target_view",
        "true_label", "predicted_label",
    }
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"Classification query table is missing columns: {sorted(missing)}")
    rows = []
    keys = ["model", "source_tma", "source_view", "target_view"]
    for identity, current in queries.groupby(keys, sort=False, observed=True):
        if len(current) < min_pair_queries or current["true_label"].nunique() < 2:
            continue
        truth = current["true_label"].astype(str)
        predicted = current["predicted_label"].astype(str)
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "queries": int(len(current)),
                "classes": int(truth.nunique()),
                "accuracy": float(np.mean(truth.to_numpy() == predicted.to_numpy())),
                "macro_f1": float(
                    f1_score(truth, predicted, average="macro", zero_division=0)
                ),
                "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            }
        )
    return pd.DataFrame(rows)


def state_transfer_queries(
    *, model: str, values: np.ndarray, frame: pd.DataFrame, state: str
) -> pd.DataFrame:
    """Transfer tumor or PD-L1 state with leave-patient-out binary centroids."""

    if state == "tumor":
        eligible = frame[frame["source_tma"].eq("tTMA1")].copy()
        eligible["state_label"] = eligible["tumor"].astype(int)
    elif state == "pdl1":
        eligible = frame[
            frame["source_tma"].eq("tTMA1") & frame["pdl1"].isin(["high", "low"])
        ].copy()
        eligible["state_label"] = eligible["pdl1"].map({"low": 0, "high": 1}).astype(int)
    else:
        raise ValueError("state must be 'tumor' or 'pdl1'")

    rows: list[dict[str, Any]] = []
    contexts = ["source_tissue", "cell_type"]
    for source_view, target_view in ordered_cross_platform_view_pairs(eligible):
        source_groups = eligible[eligible["view"].eq(source_view)].groupby(
            contexts, sort=False, observed=True
        ).groups
        target_groups = eligible[eligible["view"].eq(target_view)].groupby(
            contexts, sort=False, observed=True
        ).groups
        for context in source_groups.keys() & target_groups.keys():
            source_index = np.asarray(source_groups[context], dtype=np.int64)
            target_index = np.asarray(target_groups[context], dtype=np.int64)
            labels = eligible.loc[source_index, "state_label"].to_numpy(np.int64)
            patients = eligible.loc[source_index, "source_patient_key"].astype(str).to_numpy()
            for target_row in target_index:
                metadata = eligible.loc[int(target_row)]
                centroids = []
                for label in (0, 1):
                    keep = (labels == label) & (
                        patients != str(metadata["source_patient_key"])
                    )
                    if np.sum(keep) < 2 or np.unique(patients[keep]).size < 2:
                        centroids = []
                        break
                    centroids.append(
                        unit_rows(values[source_index[keep]].mean(axis=0, keepdims=True))[0]
                    )
                if len(centroids) != 2:
                    continue
                score = float(values[int(target_row)] @ (centroids[1] - centroids[0]))
                rows.append(
                    {
                        "model": model,
                        "state": state,
                        "source_view": source_view,
                        "target_view": target_view,
                        "source_patient_key": str(metadata["source_patient_key"]),
                        "true_label": int(metadata["state_label"]),
                        "score": score,
                        "predicted_label": int(score >= 0.0),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_state(queries: pd.DataFrame, *, min_pair_queries: int = 20) -> pd.DataFrame:
    required = {
        "model", "state", "source_view", "target_view", "source_patient_key",
        "true_label", "score", "predicted_label",
    }
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"State query table is missing columns: {sorted(missing)}")
    rows = []
    keys = ["model", "state", "source_view", "target_view"]
    for identity, current in queries.groupby(keys, sort=False, observed=True):
        if len(current) < min_pair_queries or current["true_label"].nunique() != 2:
            continue
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "queries": int(len(current)),
                "patients": int(current["source_patient_key"].nunique()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(current["true_label"], current["predicted_label"])
                ),
                "auroc": float(roc_auc_score(current["true_label"], current["score"])),
            }
        )
    return pd.DataFrame(rows)


def summarize_public_view_pairs(
    core_pairs: pd.DataFrame,
    cell_type_pairs: pd.DataFrame,
    state_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight the released directed view pairs into headline tables."""

    for name, frame, required in (
        ("core", core_pairs, {"model", *RETRIEVAL_METRICS}),
        ("cell type", cell_type_pairs, {"model", "macro_f1", "balanced_accuracy"}),
        ("state", state_pairs, {"model", "state", "balanced_accuracy"}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} view-pair table is missing: {sorted(missing)}")
    model_sets = [set(frame["model"]) for frame in (core_pairs, cell_type_pairs, state_pairs)]
    if not model_sets[0] or any(models != model_sets[0] for models in model_sets[1:]):
        raise ValueError("All independent-platform tables must contain the same models")

    retrieval = (
        core_pairs.groupby("model", as_index=False, sort=False)[list(RETRIEVAL_METRICS)]
        .mean()
    )
    transfer_rows = []
    cell_summary = cell_type_pairs.groupby("model", sort=False)[
        ["macro_f1", "balanced_accuracy"]
    ].mean()
    for metric in ("macro_f1", "balanced_accuracy"):
        values = cell_summary[metric].to_dict()
        transfer_rows.append(
            {
                "task": "cell_type",
                "metric": metric,
                **values,
                "delta": float(values["sCITEconcept"] - values["base_scConcept"]),
            }
        )
    state_summary = state_pairs.groupby(["state", "model"], sort=False)[
        "balanced_accuracy"
    ].mean()
    for state in ("tumor", "pdl1"):
        values = state_summary.loc[state].to_dict()
        transfer_rows.append(
            {
                "task": state,
                "metric": "balanced_accuracy",
                **values,
                "delta": float(values["sCITEconcept"] - values["base_scConcept"]),
            }
        )
    return retrieval, pd.DataFrame(transfer_rows)


def score_embeddings(
    metadata: pd.DataFrame,
    assignments: list[tuple[str, Path]],
    *,
    min_candidates: int = 4,
    min_pair_queries: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute all three benchmark surfaces from prototype embeddings."""

    required = {
        "prototype_id", "view", "platform", "source_tma", "source_tissue",
        "source_core_key", "source_patient_key", "tumor", "pdl1", "cell_type",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Prototype metadata is missing columns: {sorted(missing)}")
    frame = metadata.reset_index(drop=True).copy()
    if not np.array_equal(frame["prototype_id"].to_numpy(), np.arange(len(frame))):
        raise ValueError("Prototype metadata must be in prototype_id order from zero")

    core_pairs, cell_pairs, state_pairs = [], [], []
    for model, path in assignments:
        values = unit_rows(load_embeddings(path, expected_rows=len(frame)))
        core_values = residualize(
            values, frame, ["view", "source_tma", "source_tissue", "tumor", "cell_type"]
        )
        cell_values = residualize(
            values, frame, ["view", "source_tma", "source_tissue", "tumor"]
        )
        state_values = residualize(
            values, frame, ["view", "source_tma", "source_tissue", "cell_type"]
        )
        core_pairs.append(
            aggregate_core_retrieval(
                core_retrieval_queries(
                    model=model,
                    values=core_values,
                    frame=frame,
                    min_candidates=min_candidates,
                ),
                min_pair_queries=min_pair_queries,
            )
        )
        cell_pairs.append(
            aggregate_classification(
                cell_type_projection_queries(model=model, values=cell_values, frame=frame),
                min_pair_queries=min_pair_queries,
            )
        )
        current_states = pd.concat(
            [
                state_transfer_queries(
                    model=model, values=state_values, frame=frame, state=state
                )
                for state in ("tumor", "pdl1")
            ],
            ignore_index=True,
        )
        state_pairs.append(
            aggregate_state(current_states, min_pair_queries=min_pair_queries)
        )
    return (
        pd.concat(core_pairs, ignore_index=True),
        pd.concat(cell_pairs, ignore_index=True),
        pd.concat(state_pairs, ignore_index=True),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--core-view-pairs", type=Path, required=True)
    summarize.add_argument("--cell-type-view-pairs", type=Path, required=True)
    summarize.add_argument("--state-view-pairs", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--prototypes", type=Path, required=True)
    score.add_argument("--embedding", action="append", required=True, metavar="NAME=PATH")
    score.add_argument("--min-candidates", type=int, default=4)
    score.add_argument("--min-pair-queries", type=int, default=20)
    score.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "score":
        core, cell_type, state = score_embeddings(
            pd.read_csv(args.prototypes, dtype={"pdl1": str}),
            [parse_assignment(value) for value in args.embedding],
            min_candidates=args.min_candidates,
            min_pair_queries=args.min_pair_queries,
        )
        core.to_csv(args.output_dir / "independent_tma_core_view_pairs.csv", index=False)
        cell_type.to_csv(
            args.output_dir / "independent_tma_cell_type_view_pairs.csv", index=False
        )
        state.to_csv(args.output_dir / "independent_tma_state_view_pairs.csv", index=False)
    else:
        core = pd.read_csv(args.core_view_pairs)
        cell_type = pd.read_csv(args.cell_type_view_pairs)
        state = pd.read_csv(args.state_view_pairs)
    retrieval, transfer = summarize_public_view_pairs(core, cell_type, state)
    retrieval.to_csv(args.output_dir / "independent_tma_retrieval.csv", index=False)
    transfer.to_csv(args.output_dir / "independent_tma_transfer.csv", index=False)
    print(retrieval.to_string(index=False))
    print(transfer.to_string(index=False))


if __name__ == "__main__":
    main()
