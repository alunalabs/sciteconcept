#!/usr/bin/env python3
"""Rebuild every public sCITEconcept benchmark summary from row-level inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .cellchat_inhibitory_audit import audit_paired_contexts
    from .cellchat_layers import summarize_layer_scores
    from .embedding_bench import summarize_metric_rows
    from .independent_tma import summarize_public_view_pairs
    from .overall_scorecard import summarize_scorecard
except ImportError:  # direct execution
    from cellchat_inhibitory_audit import audit_paired_contexts
    from cellchat_layers import summarize_layer_scores
    from embedding_bench import summarize_metric_rows
    from independent_tma import summarize_public_view_pairs
    from overall_scorecard import summarize_scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs"))
    parser.add_argument("--audit-draws", type=int, default=100_000)
    args = parser.parse_args()
    data, output = args.data_dir, args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    pillars, leaderboard = summarize_metric_rows(
        pd.read_csv(data / "embedding_bench_metric_rows.csv")
    )
    pillars.to_csv(output / "embedding_bench_pillar_scores.csv", index=False)
    leaderboard.to_csv(output / "embedding_bench_leaderboard.csv", index=False)

    layer_scores, layer_winners = summarize_layer_scores(
        pd.read_csv(data / "cellchat_layer_scores.csv")
    )
    layer_scores.to_csv(output / "cellchat_layer_scores.csv", index=False)
    layer_winners.to_csv(output / "cellchat_family_winners.csv", index=False)

    audit = audit_paired_contexts(
        pd.read_csv(data / "cellchat_paired_contexts.csv"), draws=args.audit_draws
    )
    audit.to_csv(output / "cellchat_inhibitory_audit.csv", index=False)

    retrieval, transfer = summarize_public_view_pairs(
        pd.read_csv(data / "independent_tma_core_view_pairs.csv"),
        pd.read_csv(data / "independent_tma_cell_type_view_pairs.csv"),
        pd.read_csv(data / "independent_tma_state_view_pairs.csv"),
    )
    retrieval.to_csv(output / "independent_tma_retrieval.csv", index=False)
    transfer.to_csv(output / "independent_tma_transfer.csv", index=False)

    score_rows, score_summary, score_families = summarize_scorecard(
        pd.read_csv(data / "overall_scorecard_rows.csv")
    )
    score_rows.to_csv(output / "overall_scorecard_rows.csv", index=False)
    score_summary.to_csv(output / "overall_scorecard_summary.csv", index=False)
    score_families.to_csv(output / "overall_scorecard_family_summary.csv", index=False)

    print(leaderboard.to_string(index=False))
    print(audit.to_string(index=False))
    print(retrieval.to_string(index=False))
    print(score_summary.to_string(index=False))


if __name__ == "__main__":
    main()
