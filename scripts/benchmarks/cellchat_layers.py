#!/usr/bin/env python3
"""Score masked CellChat role embeddings across transformer layers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from ._common import load_embeddings
    from .embedding_bench import METRIC_WEIGHTS, score_mechanism
except ImportError:  # direct execution
    from _common import load_embeddings
    from embedding_bench import METRIC_WEIGHTS, score_mechanism


COMPONENT_WEIGHTS = METRIC_WEIGHTS["mechanism_pairing"]
TIE_TOLERANCE = 0.005


def mechanism_score(row: pd.Series | dict[str, Any]) -> float:
    return float(sum(weight * float(row[name]) for name, weight in COMPONENT_WEIGHTS.items()))


def summarize_layer_scores(
    layer_scores: pd.DataFrame, *, tie_tolerance: float = TIE_TOLERANCE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute weighted scores and select one best native depth per model."""

    required = {"model", "depth", "n_contexts", "n_cell_records", *COMPONENT_WEIGHTS}
    missing = required - set(layer_scores.columns)
    if missing:
        raise ValueError(f"Layer score table is missing columns: {sorted(missing)}")
    scores = layer_scores.copy()
    if scores.duplicated(["model", "depth"]).any():
        raise ValueError("Duplicate model/depth rows")
    scores["mechanism_pairing_score"] = scores.apply(mechanism_score, axis=1)
    winner_rows = []
    for model, current in scores.groupby("model", sort=False):
        ordered = current.sort_values(
            ["mechanism_pairing_score", "depth"], ascending=[False, True]
        )
        best = ordered.iloc[0]
        final = current.loc[current["depth"].eq(current["depth"].max())].iloc[0]
        delta = float(best["mechanism_pairing_score"] - final["mechanism_pairing_score"])
        winner_rows.append(
            {
                "model": model,
                "best_depth": int(best["depth"]),
                "best_score": float(best["mechanism_pairing_score"]),
                "final_layer_score": float(final["mechanism_pairing_score"]),
                "best_minus_final": delta,
                "meaningfully_better_than_final": bool(delta > tie_tolerance),
                **{name: float(best[name]) for name in COMPONENT_WEIGHTS},
            }
        )
    return scores.sort_values(["model", "depth"]), pd.DataFrame(winner_rows)


def parse_layer_embedding(value: str) -> tuple[str, int, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise ValueError(f"Expected MODEL:DEPTH=PATH, received {value!r}")
    label, raw_path = value.split("=", 1)
    model, raw_depth = label.rsplit(":", 1)
    return model, int(raw_depth), Path(raw_path).expanduser()


def score_layer_embeddings(
    records: pd.DataFrame, specifications: list[tuple[str, int, Path]]
) -> pd.DataFrame:
    rows = []
    for model, depth, path in specifications:
        embedding = load_embeddings(path, expected_rows=len(records))
        metrics, _ = score_mechanism(records, embedding)
        rows.append(
            {
                "model": model,
                "depth": depth,
                "n_contexts": int(records["context_key"].nunique()),
                "n_cell_records": int(len(records)),
                **metrics,
                "mechanism_pairing_score": sum(
                    COMPONENT_WEIGHTS[name] * metrics[name] for name in COMPONENT_WEIGHTS
                ),
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--layer-scores", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--records", type=Path, required=True)
    score.add_argument(
        "--embedding", action="append", required=True, metavar="MODEL:DEPTH=PATH"
    )
    score.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "score":
        records = pd.read_csv(args.records).reset_index(drop=True)
        initial = score_layer_embeddings(
            records, [parse_layer_embedding(value) for value in args.embedding]
        )
    else:
        initial = pd.read_csv(args.layer_scores)
    scores, winners = summarize_layer_scores(initial)
    scores.to_csv(args.output_dir / "cellchat_layer_scores.csv", index=False)
    winners.to_csv(args.output_dir / "cellchat_family_winners.csv", index=False)
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
