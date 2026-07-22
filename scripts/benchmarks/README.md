# Benchmark scripts

These programs contain the public scoring logic used by
[`sCITEconcept.ipynb`](../../sCITEconcept.ipynb). Model loading is kept outside
the metrics: use [`../encode_cells.py`](../encode_cells.py) to create NumPy
embeddings, then provide those embeddings and row-aligned metadata to a scorer.

| Program | What it computes |
| --- | --- |
| [`embedding_bench.py`](embedding_bench.py) | Frozen-benchmark pillar/composite aggregation and the masked CellChat mechanism probe |
| [`cellchat_layers.py`](cellchat_layers.py) | The same mechanism score across transformer layers and the best-depth selection rule |
| [`cellchat_inhibitory_audit.py`](cellchat_inhibitory_audit.py) | Exact paired-context construction, bootstrap interval, and paired sign-flip test |
| [`independent_tma.py`](independent_tma.py) | Cross-platform matched-core retrieval, cell-type projection, and leave-patient-out state transfer |
| [`overall_scorecard.py`](overall_scorecard.py) | Metric direction, deltas, win/loss calls, and primary/family summaries |
| [`run_all.py`](run_all.py) | Rebuild all public summaries from the checked-in row-level inputs |

Run the compact public audit from the repository root:

```bash
pip install -r requirements-benchmarks.txt
python scripts/benchmarks/run_all.py --data-dir data --output-dir benchmark_outputs
```

This command reruns score construction and included uncertainty calculations;
it does not present released outputs as fresh model inference. Fresh
embedding-level commands and their input schemas are documented in the root
README and each script's `--help` output.
