#!/usr/bin/env python3
"""Encode raw single-cell counts into sCITEconcept cell embeddings.

This is the released inference path. It reproduces exactly what produced the
published embeddings:

  * raw, non-negative counts (no log, no CPM, no scaling)
  * per cell, the genes with non-zero counts, ranked by descending count
  * truncated to ``--max-tokens`` (2048 in the release)
  * a CLS token prepended by the model
  * the CLS cell embedding, L2-normalised, float32, dimension 1024

There is no de-anisotropisation, no whitening, no global-mean subtraction and no
protein-residual sidecar. The ADT encoder was training-time only and is not used
here: sCITEconcept takes RNA counts and nothing else at inference, which is what
makes it a drop-in replacement for base scConcept.

Usage
-----
From an AnnData file whose ``.X`` holds raw counts and ``.var_names`` holds gene
symbols::

    python scripts/encode_cells.py --input cells.h5ad --output embeddings.npy

From a dense or sparse matrix plus a gene list::

    python scripts/encode_cells.py \\
        --counts counts.npy --genes genes.txt --output embeddings.npy

Both forms write a ``[cells, 1024]`` float32 array, in input row order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_sciteconcept import load_sciteconcept  # noqa: E402


DEFAULT_MAX_TOKENS = 2048
EMBEDDING_DIM = 1024


def gene_token_ids(concept, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map gene symbols onto scConcept token ids for the human vocabulary."""
    mapped = concept.map_gene_names_to_ids(species="hsapiens", gene_names=[str(g) for g in genes])
    mapped = np.asarray(mapped, dtype=object)
    token_ids = np.asarray(concept.tokenizer.encode(mapped, "hsapiens"), dtype=np.int64)
    valid = token_ids != int(concept.tokenizer.NOT_FOUND)
    return token_ids, valid


def build_batch(
    dense: np.ndarray,
    token_ids: np.ndarray,
    valid_gene: np.ndarray,
    *,
    pad_token: int,
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Rank-encode one block of cells into padded token and value matrices.

    Unlike training, which samples two random gene panels per cell, inference uses
    the deterministic full view: every expressed gene that exists in the
    vocabulary, ranked by descending count and truncated to ``max_tokens``.
    """
    dense = np.nan_to_num(np.asarray(dense, dtype=np.float32), nan=0.0, posinf=65504.0, neginf=0.0)
    dense = np.clip(dense, 0.0, 65504.0)

    rows_tokens: list[np.ndarray] = []
    rows_values: list[np.ndarray] = []
    lengths: list[int] = []
    for row in dense:
        keep = np.flatnonzero((row > 0) & valid_gene)
        values = row[keep]
        if values.size:
            order = np.argsort(-values, kind="stable")[:max_tokens]
            keep = keep[order]
            values = values[order]
            rows_tokens.append(token_ids[keep].astype(np.int64, copy=False))
            rows_values.append(values.astype(np.float32, copy=False))
        else:
            rows_tokens.append(np.empty(0, dtype=np.int64))
            rows_values.append(np.empty(0, dtype=np.float32))
        lengths.append(int(rows_tokens[-1].size))

    width = max(1, min(max_tokens, max(lengths) if lengths else 1))
    tokens = np.full((dense.shape[0], width), int(pad_token), dtype=np.int64)
    values = np.zeros((dense.shape[0], width), dtype=np.float32)
    for i, (tok, val) in enumerate(zip(rows_tokens, rows_values)):
        take = min(width, int(tok.size))
        if take:
            tokens[i, :take] = tok[:take]
            values[i, :take] = val[:take]
        lengths[i] = take
    return tokens, values, lengths


@torch.no_grad()
def encode_block(model, tokens, values, lengths, *, device: str) -> torch.Tensor:
    """Run one padded block through the encoder and return normalised CLS embeddings."""
    model.stage = "val"
    model.LOGGING_STEP = False
    model.set_active_species("hsapiens")
    batch = {
        "tokens": torch.as_tensor(tokens, device=device, dtype=torch.long),
        "values": torch.as_tensor(values, device=device, dtype=torch.float32),
        "seq_lengths": [int(x) for x in lengths],
    }
    batch = model.add_cls_token(batch)
    _pred, _embs, cell_embs = model(batch["tokens"], batch["values"], seq_lengths=batch["seq_lengths"])
    if getattr(model, "projection_dim", None):
        cell_embs = model.projection(cell_embs)
    return F.normalize(cell_embs.float(), p=2, dim=1)


def load_counts(args) -> tuple[np.ndarray, list[str]]:
    """Return a dense count matrix and its gene symbols."""
    if args.input is not None:
        import anndata

        adata = anndata.read_h5ad(args.input)
        matrix = adata.X
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        return matrix, [str(g) for g in adata.var_names]

    matrix = np.load(args.counts, allow_pickle=False)
    if matrix.ndim != 2:
        raise ValueError(f"--counts must be a 2-D [cells, genes] array, got shape {matrix.shape}")
    genes = [line.strip() for line in Path(args.genes).read_text().splitlines() if line.strip()]
    if len(genes) != matrix.shape[1]:
        raise ValueError(f"--genes has {len(genes)} entries but --counts has {matrix.shape[1]} columns")
    return matrix, genes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="encode_cells.py",
        description="Encode raw counts into sCITEconcept cell embeddings.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="AnnData .h5ad with raw counts in .X.")
    source.add_argument("--counts", type=Path, help="Dense [cells, genes] .npy of raw counts.")
    parser.add_argument("--genes", type=Path, help="Gene symbols, one per line. Required with --counts.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .npy for the embeddings.")
    parser.add_argument("--batch-size", type=int, default=32, help="Cells encoded per forward pass.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Ranked genes kept per cell (release used 2048).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Where to cache downloaded weights.")
    parser.add_argument("--token", default=None, help="Hugging Face token, if the repo is private.")
    parser.add_argument("--write-metadata", action="store_true",
                        help="Also write <output>.json describing the run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.counts is not None and args.genes is None:
        raise SystemExit("--genes is required when using --counts")

    matrix, genes = load_counts(args)
    if matrix.min() < 0:
        raise SystemExit("counts are negative; this model expects raw, non-negative counts")
    print(f"Loaded {matrix.shape[0]:,} cells x {matrix.shape[1]:,} genes", flush=True)

    concept, metadata = load_sciteconcept(cache_dir=args.cache_dir, token=args.token,
                                          map_location=args.device)
    token_ids, valid_gene = gene_token_ids(concept, genes)
    print(f"{int(valid_gene.sum()):,} of {len(genes):,} genes map into the scConcept vocabulary",
          flush=True)
    if not valid_gene.any():
        raise SystemExit("no genes mapped; check that --genes holds human gene symbols")

    pad_token = int(concept.tokenizer.PAD_TOKEN)
    out = np.empty((matrix.shape[0], EMBEDDING_DIM), dtype=np.float32)
    for start in range(0, matrix.shape[0], args.batch_size):
        stop = min(start + args.batch_size, matrix.shape[0])
        tokens, values, lengths = build_batch(
            matrix[start:stop], token_ids, valid_gene,
            pad_token=pad_token, max_tokens=args.max_tokens,
        )
        out[start:stop] = encode_block(
            concept.model, tokens, values, lengths, device=args.device
        ).cpu().numpy()
        print(f"  encoded {stop:,}/{matrix.shape[0]:,}", end="\r", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"\nWrote {out.shape} float32 embeddings to {args.output}")

    if args.write_metadata:
        side = args.output.with_suffix(args.output.suffix + ".json")
        side.write_text(json.dumps({
            "n_cells": int(out.shape[0]),
            "embedding_dim": int(out.shape[1]),
            "max_tokens": int(args.max_tokens),
            "genes_mapped": int(valid_gene.sum()),
            "genes_supplied": int(len(genes)),
            "input_space": "raw_counts_rank_encoding",
            "postprocessing": "none; L2-normalised CLS cell embedding",
            "checkpoint": metadata,
        }, indent=2) + "\n")
        print(f"Wrote run metadata to {side}")


if __name__ == "__main__":
    main()
