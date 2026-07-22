#!/usr/bin/env python3
"""Validate and package aligned CITE-seq arrays for sCITEconcept training.

The input arrays must already use a shared gene and protein vocabulary. This
script performs no dataset-specific downloading or parsing; it documents the
portable boundary between source-data preparation and model training.

Required inputs
---------------
rna.npy
    Raw, non-negative RNA counts with shape ``[cells, genes]``.
protein.npy
    Protein measurements with shape ``[cells, proteins]``.
protein_mask.npy
    Boolean array with the same shape as ``protein.npy``. ``True`` means the
    protein was measured for that cell's source dataset.
split.npy
    Integer vector with one value per cell: 0 for training, 1 for validation.
dataset_id.npy
    Integer vector identifying each source dataset. Training batches are kept
    within one dataset.
genes.txt / proteins.txt
    One feature name per line, in array-column order.

The output is the memory-mappable directory consumed by
``train_sciteconcept.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


ARRAY_FILES = {
    "rna": "rna.npy",
    "protein": "protein.npy",
    "protein_mask": "protein_mask.npy",
    "split": "split.npy",
    "dataset_id": "dataset_id.npy",
}


def read_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"feature list is empty: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"feature names must be unique: {path}")
    return names


def validate_inputs(input_dir: Path) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    arrays = {
        key: np.load(input_dir / filename, mmap_mode="r")
        for key, filename in ARRAY_FILES.items()
    }
    genes = read_names(input_dir / "genes.txt")
    proteins = read_names(input_dir / "proteins.txt")

    rna = arrays["rna"]
    protein = arrays["protein"]
    mask = arrays["protein_mask"]
    split = arrays["split"]
    dataset_id = arrays["dataset_id"]

    if rna.ndim != 2 or protein.ndim != 2:
        raise ValueError("RNA and protein arrays must both be two-dimensional")
    if rna.shape[0] == 0:
        raise ValueError("the input arrays contain no cells")
    if rna.shape[0] != protein.shape[0]:
        raise ValueError("RNA and protein arrays must contain the same cells in the same order")
    if mask.shape != protein.shape:
        raise ValueError("protein_mask must have the same shape as protein")
    if mask.dtype != np.bool_:
        raise ValueError("protein_mask must use a boolean dtype")
    if split.shape != (rna.shape[0],) or dataset_id.shape != (rna.shape[0],):
        raise ValueError("split and dataset_id must be one-dimensional cell vectors")
    if rna.shape[1] != len(genes) or protein.shape[1] != len(proteins):
        raise ValueError("feature lists do not match the matrix column counts")
    if not np.isin(np.asarray(split), [0, 1]).all():
        raise ValueError("split must contain only 0 (train) and 1 (validation)")
    if set(np.unique(np.asarray(split)).tolist()) != {0, 1}:
        raise ValueError("split must contain at least one training and one validation cell")
    if not np.issubdtype(dataset_id.dtype, np.integer):
        raise ValueError("dataset_id must use an integer dtype")

    for start in range(0, rna.shape[0], 4096):
        stop = min(start + 4096, rna.shape[0])
        rna_chunk = np.asarray(rna[start:stop], dtype=np.float32)
        protein_chunk = np.asarray(protein[start:stop], dtype=np.float32)
        if not np.isfinite(rna_chunk).all() or (rna_chunk < 0).any():
            raise ValueError(f"RNA must contain finite, non-negative raw counts; failed rows {start}:{stop}")
        measured = np.asarray(mask[start:stop], dtype=bool)
        if not np.isfinite(protein_chunk[measured]).all():
            raise ValueError(f"measured protein values must be finite; failed rows {start}:{stop}")

    return arrays, genes, proteins


def copy_array(source: np.ndarray, destination: Path, *, chunk_rows: int) -> None:
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=source.dtype,
        shape=source.shape,
    )
    rows = source.shape[0] if source.ndim else 1
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        output[start:stop] = source[start:stop]
    output.flush()


def build_pack(input_dir: Path, output_dir: Path, *, chunk_rows: int) -> dict:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input_dir and output_dir must be different directories")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")

    arrays, genes, proteins = validate_inputs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in ARRAY_FILES.items():
        copy_array(arrays[key], output_dir / filename, chunk_rows=chunk_rows)

    for name in ("genes.txt", "proteins.txt"):
        shutil.copy2(input_dir / name, output_dir / name)

    dataset_names_path = input_dir / "datasets.json"
    if dataset_names_path.exists():
        dataset_names = json.loads(dataset_names_path.read_text(encoding="utf-8"))
        shutil.copy2(dataset_names_path, output_dir / "datasets.json")
    else:
        ids = sorted(int(value) for value in np.unique(arrays["dataset_id"]))
        dataset_names = {str(value): f"dataset_{value}" for value in ids}
        (output_dir / "datasets.json").write_text(
            json.dumps(dataset_names, indent=2) + "\n",
            encoding="utf-8",
        )

    split = np.asarray(arrays["split"])
    mask = arrays["protein_mask"]
    observed_counts = np.empty(mask.shape[0], dtype=np.int32)
    for start in range(0, mask.shape[0], chunk_rows):
        stop = min(start + chunk_rows, mask.shape[0])
        observed_counts[start:stop] = np.asarray(mask[start:stop], dtype=bool).sum(axis=1)
    metadata = {
        "format": "sciteconcept-training-pack-v1",
        "n_cells": int(arrays["rna"].shape[0]),
        "n_genes": int(arrays["rna"].shape[1]),
        "n_proteins": int(arrays["protein"].shape[1]),
        "genes": genes,
        "proteins": proteins,
        "datasets": dataset_names,
        "split_counts": {
            "train": int((split == 0).sum()),
            "validation": int((split == 1).sum()),
        },
        "observed_proteins_per_cell": {
            "minimum": int(observed_counts.min()),
            "median": float(np.median(observed_counts)),
            "maximum": int(observed_counts.max()),
        },
        "rna_input": "raw non-negative counts",
        "files": ARRAY_FILES,
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_pack(args.input_dir, args.output_dir, chunk_rows=args.chunk_rows)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
