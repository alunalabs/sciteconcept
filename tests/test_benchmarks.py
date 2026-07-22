from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmarks.cellchat_inhibitory_audit import audit_paired_contexts
from scripts.benchmarks.cellchat_layers import summarize_layer_scores
from scripts.benchmarks.embedding_bench import summarize_metric_rows
from scripts.benchmarks.independent_tma import summarize_public_view_pairs
from scripts.benchmarks.overall_scorecard import summarize_scorecard


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_public_benchmark_reconstructions() -> None:
    pillars, leaderboard = summarize_metric_rows(
        pd.read_csv(DATA / "embedding_bench_metric_rows.csv")
    )
    recorded_pillars = pd.read_csv(DATA / "embedding_bench_pillar_scores.csv")
    recorded_leaderboard = pd.read_csv(DATA / "embedding_bench_leaderboard.csv")
    assert np.allclose(
        pillars.sort_values(["model", "pillar"])["pillar_score"],
        recorded_pillars.sort_values(["model", "pillar"])["pillar_score"],
        atol=1e-12,
    )
    assert np.allclose(
        leaderboard.sort_values("model")["representation_score"],
        recorded_leaderboard.sort_values("model")["representation_score"],
        atol=1e-12,
    )

    rebuilt_rows, summary, families = summarize_scorecard(
        pd.read_csv(DATA / "overall_scorecard_rows.csv")
    )
    assert summary.iloc[0][["primary_metrics", "wins", "losses", "ties"]].tolist() == [
        60,
        49,
        11,
        0,
    ]
    assert len(rebuilt_rows) == 75
    assert len(families) == 7

    layers, winners = summarize_layer_scores(
        pd.read_csv(DATA / "cellchat_layer_scores.csv")
    )
    assert np.allclose(
        layers["mechanism_pairing_score"],
        pd.read_csv(DATA / "cellchat_layer_scores.csv")["mechanism_pairing_score"],
        atol=1e-12,
    )
    assert winners.set_index("model")["best_depth"].to_dict() == {
        "base_scConcept": 6,
        "sCITEconcept": 13,
    }


def test_public_row_level_audits() -> None:
    retrieval, transfer = summarize_public_view_pairs(
        pd.read_csv(DATA / "independent_tma_core_view_pairs.csv"),
        pd.read_csv(DATA / "independent_tma_cell_type_view_pairs.csv"),
        pd.read_csv(DATA / "independent_tma_state_view_pairs.csv"),
    )
    recorded_retrieval = pd.read_csv(DATA / "independent_tma_retrieval.csv")
    recorded_transfer = pd.read_csv(DATA / "independent_tma_transfer.csv")
    for column in ["pairwise_auc", "mrr", "recall_at_1", "recall_at_5"]:
        assert np.allclose(
            retrieval.sort_values("model")[column],
            recorded_retrieval.sort_values("model")[column],
            atol=5e-5,
        )
    for column in ["sCITEconcept", "base_scConcept", "delta"]:
        assert np.allclose(transfer[column], recorded_transfer[column], atol=1e-4)

    audit = audit_paired_contexts(
        pd.read_csv(DATA / "cellchat_paired_contexts.csv"),
        draws=100_000,
        seed=20260709,
    ).set_index("score")
    inhibitory = audit.loc["inhibitory_pass_rate"]
    assert int(inhibitory["n_contexts"]) == 42
    assert round(float(inhibitory["p_value"]), 3) == 0.901
    assert float(inhibitory["ci_low_points"]) < 0 < float(inhibitory["ci_high_points"])
