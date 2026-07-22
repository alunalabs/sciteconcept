#!/usr/bin/env python3
"""Run the paired inhibitory/coexpressive CellChat audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from ._common import as_bool
except ImportError:  # direct execution
    from _common import as_bool


PAIR_COUNT_FLOOR = 40
BOOTSTRAP_DRAWS = 100_000
SEED = 20260709


def build_paired_contexts(
    checkpoint_rows: pd.DataFrame,
    *,
    scite_model: str,
    base_model: str,
    pair_count_floor: int = PAIR_COUNT_FLOOR,
) -> pd.DataFrame:
    """Reduce model/checkpoint direction rows to exact paired context rates.

    Required columns are ``model``, ``checkpoint``, ``row_id``,
    ``relationship_class``, ``status``, ``direction_pass``, and ``pair_count``.
    The upstream model stack is intentionally abstracted to this row contract.
    """

    required = {
        "model",
        "checkpoint",
        "row_id",
        "relationship_class",
        "status",
        "direction_pass",
        "pair_count",
    }
    missing = required - set(checkpoint_rows.columns)
    if missing:
        raise ValueError(f"Checkpoint table is missing columns: {sorted(missing)}")
    frame = checkpoint_rows.copy()
    frame["pair_count"] = pd.to_numeric(frame["pair_count"], errors="raise")
    frame = frame[
        frame["model"].isin([scite_model, base_model])
        & frame["status"].astype(str).str.lower().eq("ok")
        & frame["direction_pass"].notna()
        & frame["pair_count"].ge(pair_count_floor)
    ].copy()
    frame["pass"] = frame["direction_pass"].map(as_bool).astype(float)
    rates = (
        frame.groupby(["model", "row_id", "relationship_class"], observed=True)
        .agg(pass_rate=("pass", "mean"), checkpoints=("checkpoint", "nunique"), pair_count=("pair_count", "max"))
        .reset_index()
    )
    identity = ["row_id", "relationship_class", "pair_count"]
    scite = rates[rates["model"].eq(scite_model)].drop(columns="model")
    base = rates[rates["model"].eq(base_model)].drop(columns="model")
    paired = scite.merge(
        base,
        on=identity,
        suffixes=("_sciteconcept", "_base_scconcept"),
        validate="one_to_one",
    )
    paired["delta_pass_rate"] = (
        paired["pass_rate_sciteconcept"] - paired["pass_rate_base_scconcept"]
    )
    return paired.sort_values(["relationship_class", "row_id"]).reset_index(drop=True)


def paired_test(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, float | int]:
    """Paired context bootstrap interval and two-sided sign-flip test."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Paired deltas must be a finite one-dimensional vector")
    if draws <= 0:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    null = (
        rng.choice(np.array([-1.0, 1.0]), size=(draws, len(values))) * values
    ).mean(axis=1)
    observed = float(values.mean())
    return {
        "n_contexts": int(len(values)),
        "mean_delta": observed,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "p_value": float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + draws)),
        "sciteconcept_wins": int(np.sum(values > 0)),
        "base_scconcept_wins": int(np.sum(values < 0)),
        "ties": int(np.sum(values == 0)),
    }


def audit_paired_contexts(
    paired_contexts: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return balanced, inhibitory, and coexpressive public audit rows."""

    aliases = {
        "pass_rate_scite": "sciteconcept_pass_rate",
        "pass_rate_base": "base_scconcept_pass_rate",
        "pass_rate_sciteconcept": "sciteconcept_pass_rate",
        "pass_rate_base_scconcept": "base_scconcept_pass_rate",
    }
    frame = paired_contexts.rename(columns=aliases).copy()
    required = {
        "row_id",
        "relationship_class",
        "sciteconcept_pass_rate",
        "base_scconcept_pass_rate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Paired table is missing columns: {sorted(missing)}")
    for column in ["sciteconcept_pass_rate", "base_scconcept_pass_rate"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    identity = "context_key" if "context_key" in frame.columns else "row_id"
    if frame.duplicated([identity]).any():
        raise ValueError("Paired context identifiers must be unique")
    frame["delta_pass_rate"] = (
        frame["sciteconcept_pass_rate"] - frame["base_scconcept_pass_rate"]
    )

    class_means = frame.groupby("relationship_class", observed=True)[
        ["sciteconcept_pass_rate", "base_scconcept_pass_rate"]
    ].mean()
    if set(class_means.index) != {"coexpressive", "inhibitory"}:
        raise ValueError("Both coexpressive and inhibitory contexts are required")
    balanced_scite = float(class_means["sciteconcept_pass_rate"].mean())
    balanced_base = float(class_means["base_scconcept_pass_rate"].mean())
    rows: list[dict[str, Any]] = [
        {
            "score": "balanced_pass_rate",
            "sciteconcept": balanced_scite,
            "base_scconcept": balanced_base,
            "delta_points": 100.0 * (balanced_scite - balanced_base),
            "ci_low_points": np.nan,
            "ci_high_points": np.nan,
            "p_value": np.nan,
            "n_contexts": int(len(frame)),
            "interpretation": "Overall stack-level family difference",
        }
    ]
    for offset, relationship in enumerate(["coexpressive", "inhibitory"]):
        current = frame[frame["relationship_class"].eq(relationship)]
        test = paired_test(
            current["delta_pass_rate"].to_numpy(), seed=seed + offset, draws=draws
        )
        rows.append(
            {
                "score": f"{relationship}_pass_rate",
                "sciteconcept": float(current["sciteconcept_pass_rate"].mean()),
                "base_scconcept": float(current["base_scconcept_pass_rate"].mean()),
                "delta_points": 100.0 * float(test["mean_delta"]),
                "ci_low_points": 100.0 * float(test["ci_low"]),
                "ci_high_points": 100.0 * float(test["ci_high"]),
                "p_value": float(test["p_value"]) if relationship == "inhibitory" else np.nan,
                "n_contexts": int(test["n_contexts"]),
                "interpretation": (
                    "Statistical tie" if relationship == "inhibitory" else "Source of the balanced lift"
                ),
            }
        )
    order = {"balanced_pass_rate": 0, "inhibitory_pass_rate": 1, "coexpressive_pass_rate": 2}
    return pd.DataFrame(rows).sort_values("score", key=lambda values: values.map(order)).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--paired-contexts", type=Path)
    source.add_argument("--checkpoint-rows", type=Path)
    parser.add_argument("--scite-model", default="sCITEconcept")
    parser.add_argument("--base-model", default="base_scConcept")
    parser.add_argument("--pair-count-floor", type=int, default=PAIR_COUNT_FLOOR)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.paired_contexts is not None:
        paired = pd.read_csv(args.paired_contexts)
    else:
        paired = build_paired_contexts(
            pd.read_csv(args.checkpoint_rows),
            scite_model=args.scite_model,
            base_model=args.base_model,
            pair_count_floor=args.pair_count_floor,
        )
        paired.to_csv(args.output.with_name("cellchat_paired_contexts.csv"), index=False)
    audit = audit_paired_contexts(paired, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
