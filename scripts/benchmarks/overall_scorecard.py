#!/usr/bin/env python3
"""Validate and rebuild the broad sCITEconcept-versus-base scorecard.

This is an aggregation script, not model inference. It consumes one row per
already-scored metric, applies the declared higher/lower direction, and
recomputes deltas, outcomes, the primary-metric summary, and family summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ._common import as_bool
except ImportError:  # direct execution
    from _common import as_bool


def rebuild_scorecard_rows(rows: pd.DataFrame, *, tolerance: float = 1.0e-12) -> pd.DataFrame:
    required = {
        "family", "benchmark", "metric", "primary", "direction",
        "sCITEconcept", "base_scConcept",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Scorecard is missing columns: {sorted(missing)}")
    frame = rows.copy()
    if frame.duplicated(["family", "benchmark", "metric"]).any():
        raise ValueError("Scorecard metric identities must be unique")
    frame["primary"] = frame["primary"].map(as_bool)
    frame["sCITEconcept"] = pd.to_numeric(frame["sCITEconcept"], errors="raise")
    frame["base_scConcept"] = pd.to_numeric(frame["base_scConcept"], errors="raise")
    if not np.isfinite(frame[["sCITEconcept", "base_scConcept"]].to_numpy()).all():
        raise ValueError("Scorecard values must be finite")
    if not set(frame["direction"]).issubset({"higher", "lower"}):
        raise ValueError("Metric direction must be 'higher' or 'lower'")

    frame["raw_delta"] = frame["sCITEconcept"] - frame["base_scConcept"]
    frame["directional_delta"] = np.where(
        frame["direction"].eq("higher"), frame["raw_delta"], -frame["raw_delta"]
    )
    denominator = frame["base_scConcept"].abs()
    frame["directional_relative_delta"] = np.where(
        denominator.gt(tolerance), frame["directional_delta"] / denominator, np.nan
    )
    frame["result"] = np.select(
        [
            frame["directional_delta"].gt(tolerance),
            frame["directional_delta"].lt(-tolerance),
        ],
        ["win", "loss"],
        default="tie",
    )
    preferred = [
        "family", "benchmark", "metric", "primary", "direction",
        "sCITEconcept", "base_scConcept", "raw_delta", "directional_delta",
        "directional_relative_delta", "result", "note",
    ]
    return frame[[column for column in preferred if column in frame.columns]]


def summarize_scorecard(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return rebuilt metric rows, the overall summary, and family summary."""

    rebuilt = rebuild_scorecard_rows(rows)
    primary = rebuilt[rebuilt["primary"]].copy()
    counts = primary["result"].value_counts()
    summary = pd.DataFrame(
        [
            {
                "comparison": "sCITEconcept vs base scConcept",
                "primary_metrics": int(len(primary)),
                "wins": int(counts.get("win", 0)),
                "losses": int(counts.get("loss", 0)),
                "ties": int(counts.get("tie", 0)),
                "win_rate": float(counts.get("win", 0) / len(primary)),
            }
        ]
    )
    family_rows = []
    for family, current in primary.groupby("family", sort=False):
        current_counts = current["result"].value_counts()
        family_rows.append(
            {
                "family": family,
                "primary_metrics": int(len(current)),
                "wins": int(current_counts.get("win", 0)),
                "losses": int(current_counts.get("loss", 0)),
                "ties": int(current_counts.get("tie", 0)),
                "win_rate": float(current_counts.get("win", 0) / len(current)),
                "median_directional_delta": float(current["directional_delta"].median()),
                "mean_directional_delta": float(current["directional_delta"].mean()),
            }
        )
    families = pd.DataFrame(family_rows).sort_values(
        ["win_rate", "family"], ascending=[False, True]
    ).reset_index(drop=True)
    return rebuilt, summary, families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rebuilt, summary, families = summarize_scorecard(pd.read_csv(args.rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(args.output_dir / "overall_scorecard_rows.csv", index=False)
    summary.to_csv(args.output_dir / "overall_scorecard_summary.csv", index=False)
    families.to_csv(args.output_dir / "overall_scorecard_family_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
