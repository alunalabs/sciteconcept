# sCITEconcept

sCITEconcept is an RNA encoder obtained by fine-tuning
[scConcept](https://huggingface.co/theislab/scConcept) on paired CITE-seq RNA
and surface-protein measurements. Protein measurements are used only during
training; inference requires RNA counts alone.

The executed **[`sCITEconcept.ipynb`](sCITEconcept.ipynb)** is the main entry
point. It explains the training method, defines every benchmark and score,
calls the public scoring functions against the included row-level inputs, and
states the main limitations next to the results they qualify. GitHub renders
the notebook with its outputs, so no local setup is needed to read it.

## What is in this repository

| Path | Contents |
| --- | --- |
| [`sCITEconcept.ipynb`](sCITEconcept.ipynb) | Executed training and benchmark walkthrough |
| [`scripts/build_training_pack.py`](scripts/build_training_pack.py) | Validates aligned RNA/protein arrays and creates the training input pack |
| [`scripts/train_sciteconcept.py`](scripts/train_sciteconcept.py) | Portable local training program used by the documented recipe |
| [`scripts/objective.py`](scripts/objective.py) | Compact, readable implementation of the three-view contrastive loss |
| [`scripts/load_sciteconcept.py`](scripts/load_sciteconcept.py) | Loads the released checkpoint on top of the pinned base model |
| [`scripts/encode_cells.py`](scripts/encode_cells.py) | Command-line encoder: raw counts in, 1024-dimensional cell embeddings out |
| [`scripts/benchmarks/`](scripts/benchmarks) | Executable benchmark scoring, aggregation, and audit programs |
| [`data/`](data) | Training records plus row-level and summary benchmark tables used by the notebook |
| [`requirements-benchmarks.txt`](requirements-benchmarks.txt) | Notebook and benchmark dependencies |
| [`NOTICE.md`](NOTICE.md) | Upstream model and data attribution |
| [`LICENSE`](LICENSE) | MIT license |

The notebook is the methodology document as well as the result audit. The CSV
files keep each reported result inspectable, while the scripts contain the
corresponding data contract, objective, training loop, and inference loader.

## Read or run the notebook

To execute the notebook locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-benchmarks.txt
jupyter execute sCITEconcept.ipynb --inplace
```

The notebook reruns the public score construction and statistical audits from
small committed inputs under `data/`; it does not download a model or require a
GPU. Fresh encoding and the full embedding-level benchmarks are separate
commands because source assays are not redistributed here.

## Reproduce the training recipe

Source CITE-seq datasets must first be harmonized into aligned arrays. Dataset
download and source-specific parsing are not included because source formats
and redistribution terms differ. The public input boundary is:

| File | Required contents |
| --- | --- |
| `rna.npy` | Non-negative raw RNA counts, shaped `[cells, genes]` |
| `protein.npy` | Protein measurements, shaped `[cells, proteins]` |
| `protein_mask.npy` | Boolean measured-value mask with the same shape as `protein.npy` |
| `split.npy` | Cell-level split: `0` for training and `1` for validation |
| `dataset_id.npy` | Integer source-dataset identifier for each cell |
| `genes.txt`, `proteins.txt` | One feature name per line, in matrix-column order |
| `datasets.json` | Optional mapping from dataset identifiers to names |

After installing PyTorch, NumPy, Safetensors, `huggingface_hub`, OmegaConf, and
scConcept 0.2.5, validate the arrays and start the first training segment
locally:

```bash
python scripts/build_training_pack.py \
  --input-dir prepared_arrays \
  --output-dir training_pack

python scripts/train_sciteconcept.py \
  --pack-dir training_pack \
  --output-dir training_run \
  --config data/training_config.json \
  --epochs 4
```

The trainer downloads the pinned base scConcept model when it is not already
cached. Set `SCITECONCEPT_MODEL_CACHE` to choose the base-model cache location.
It writes local checkpoints and a JSON report under the requested output
directory. The released trajectory used four training segments; each
continuation restored model weights and began with a fresh AdamW optimizer, as
documented in the notebook and `data/training_trajectory.csv`.

## Use the released model

[`scripts/load_sciteconcept.py`](scripts/load_sciteconcept.py) retrieves
`sCITEconcept.safetensors` from
[`alunalabs/sCITEconcept`](https://huggingface.co/alunalabs/sCITEconcept), loads
the pinned `theislab/scConcept` base artifact, and applies the fine-tuned RNA
encoder weights. The returned model produces 1,024-dimensional cell
embeddings. No protein encoder, sidecar, whitening, centering, or other
post-processing is used at inference time.

**Checkpoint publication status (July 21, 2026):** the organization model page
has not yet been created, so automatic download from that release ID is not
available today. The loader already targets the intended stable ID. A local
release bundle can be used immediately with `--weights` and `--metadata`; the
public page requires an `alunalabs` maintainer with model-repository creation
permission.

To encode your own cells, use `scripts/encode_cells.py`. It expects **raw,
non-negative counts** and human gene symbols:

```bash
python scripts/encode_cells.py \
    --input cells.h5ad \
    --output embeddings.npy \
    --write-metadata
```

With a local release bundle:

```bash
python scripts/encode_cells.py \
  --input cells.h5ad --output embeddings.npy \
  --weights sCITEconcept.safetensors \
  --metadata checkpoint_metadata.json
```

It writes a `[cells, 1024]` float32 array in input row order. Per cell, the
expressed genes are ranked by descending count and truncated to 2,048 tokens,
which is the contract the model was trained under. Do not log-transform or
CPM-normalise first: that changes the ranking and silently degrades the
embedding. Nothing is applied on top of the L2-normalised CLS output.

## Run the benchmarks

The public benchmark boundary is an embedding matrix plus metadata in the same
row order. This keeps model loading replaceable while making the evaluation
logic inspectable. The released encoder command above produces the embedding
matrix; comparator encoders can write the same NumPy format.

Rebuild every table-level public summary and paired statistical audit:

```bash
python scripts/benchmarks/run_all.py \
  --data-dir data \
  --output-dir benchmark_outputs
```

Run the masked sender/receiver mechanism probe from fresh cell embeddings. The
records CSV has one row per embedded cell and includes `context_key`, `role`,
`interaction`, `relationship`, and `tissue`:

```bash
python scripts/benchmarks/embedding_bench.py mechanism \
  --records mechanism_records.csv \
  --embedding sCITEconcept=scite_embeddings.npy \
  --embedding base_scConcept=base_embeddings.npy \
  --output benchmark_outputs/mechanism_metric_rows.csv
```

The same records and embeddings can be scored layer by layer:

```bash
python scripts/benchmarks/cellchat_layers.py score \
  --records mechanism_records.csv \
  --embedding sCITEconcept:13=scite_layer13.npy \
  --embedding sCITEconcept:16=scite_layer16.npy \
  --embedding base_scConcept:16=base_layer16.npy \
  --output-dir benchmark_outputs/layers
```

Run the independent spatial-platform benchmark from prototype embeddings. The
metadata schema and all three scoring surfaces—matched-core retrieval,
leave-patient-out cell-type projection, and tumor/PD-L1 state transfer—are
implemented in the referenced script:

```bash
python scripts/benchmarks/independent_tma.py score \
  --prototypes prototypes.csv \
  --embedding sCITEconcept=scite_prototypes.npy \
  --embedding base_scConcept=base_prototypes.npy \
  --output-dir benchmark_outputs/independent_tma
```

Raw study matrices and study-specific parsers are not redistributed. The
checked-in row-level tables let the notebook and `run_all.py` rerun every
reported aggregation, direction rule, and included uncertainty calculation;
fresh embedding-level runs require the corresponding source-study inputs.

## How to interpret the results

The central comparison is sCITEconcept versus the unchanged base scConcept
encoder. The notebook defines the unit, metric direction, aggregation, and
weight for every score before interpreting it. In brief, sCITEconcept improves
several cell-level transfer and cross-platform retrieval results, while the
overall frozen-embedding composite gain is small and does not reach its stated
winner margin. Base scConcept remains stronger on native-panel invariance and
some gene-level tasks. Cell-communication gains are clearest for coexpressive
relationships, while the inhibitory comparison is statistically unresolved.

## License and citation

Repository-authored code is available under the MIT license. Upstream models
and datasets retain their own terms; see [`NOTICE.md`](NOTICE.md). Citation
metadata is provided in [`CITATION.cff`](CITATION.cff).
