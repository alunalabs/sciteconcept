# sCITEconcept

sCITEconcept is an RNA encoder obtained by fine-tuning
[scConcept](https://huggingface.co/theislab/scConcept) on paired CITE-seq RNA
and surface-protein measurements. Protein measurements are used only during
training; inference requires RNA counts alone.

The executed **[`sCITEconcept.ipynb`](sCITEconcept.ipynb)** is the main entry
point. It explains the training method, defines every benchmark and score,
reconstructs the reported summaries from the included result tables, and
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
| [`data/`](data) | Training history, configuration, model metadata, and benchmark result tables |
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
pip install jupyter pandas matplotlib
jupyter execute sCITEconcept.ipynb --inplace
```

The notebook reads only the small files committed under `data/`; it does not
download a model or require a GPU.

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
