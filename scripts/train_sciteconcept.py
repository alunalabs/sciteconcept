#!/usr/bin/env python3
"""Fine-tune scConcept with paired CITE-seq RNA and protein views.

This is a portable version of the training program used for sCITEconcept v1.
It consumes the array pack documented by ``build_training_pack.py`` and writes
local checkpoints and a JSON training report.

Important historical detail: optimizer state was not part of the released
checkpoint format. Resuming restores the scConcept and ADT encoder weights and
starts a fresh AdamW optimizer, matching the four-segment release trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


TRAINING_RECIPE = "sciteconcept-v1"
SC_REPO_ID = "theislab/scConcept"
SC_MODEL_NAME = "corpus230M[human]-model170M"
SC_REVISION = "10ed3ec8f35249247c33e1835e381a4c935ee26f"
SC_MODEL_GLOB = "corpus230M[[]human[]]-model170M/**"
MODEL_CACHE_ROOT = Path(
    os.environ.get("SCITECONCEPT_MODEL_CACHE", Path.home() / ".cache" / "sciteconcept" / "base_model")
).expanduser()

def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)

def _log(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, default=_json_default), flush=True)

def _stats(values: Any) -> dict[str, Any]:
    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{int(time.time() * 1000000)}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)

def _load_pack(pack_dir: Path):
    import numpy as np

    meta = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
    rna = np.load(pack_dir / meta["files"]["rna"], mmap_mode="r")
    protein = np.load(pack_dir / meta["files"]["protein"], mmap_mode="r")
    mask = np.load(pack_dir / meta["files"]["protein_mask"], mmap_mode="r")
    split = np.load(pack_dir / meta["files"]["split"], mmap_mode="r")
    dataset_id = np.load(pack_dir / meta["files"]["dataset_id"], mmap_mode="r")
    return meta, rna, protein, mask, split, dataset_id

def _observed_counts(mask, *, chunk_rows: int = 65536):
    import numpy as np

    counts = np.empty(mask.shape[0], dtype=np.int32)
    for start in range(0, int(mask.shape[0]), int(chunk_rows)):
        stop = min(start + int(chunk_rows), int(mask.shape[0]))
        counts[start:stop] = np.asarray(mask[start:stop], dtype=bool).sum(axis=1).astype(np.int32, copy=False)
    return counts

def _protein_norm_stats(protein, mask, rows, *, chunk_rows: int):
    import numpy as np

    n_proteins = int(protein.shape[1])
    count = np.zeros(n_proteins, dtype=np.float64)
    sum_x = np.zeros(n_proteins, dtype=np.float64)
    sum_x2 = np.zeros(n_proteins, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    for start in range(0, rows.size, int(chunk_rows)):
        stop = min(start + int(chunk_rows), rows.size)
        idx = rows[start:stop]
        y = np.asarray(protein[idx], dtype=np.float32)
        m = np.asarray(mask[idx], dtype=bool)
        w = m.astype(np.float64)
        y64 = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        count += w.sum(axis=0)
        sum_x += (y64 * w).sum(axis=0)
        sum_x2 += ((y64 * y64) * w).sum(axis=0)
    mean = np.divide(sum_x, np.maximum(count, 1.0), out=np.zeros_like(sum_x), where=count > 0)
    var = np.divide(sum_x2, np.maximum(count, 1.0), out=np.zeros_like(sum_x2), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(var, 1.0e-6))
    std[count < 2] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "count": count.astype(np.int64)}

def _select_rows(split, dataset_id, eligible, *, max_rows: int, seed: int):
    import numpy as np

    eligible = np.asarray(eligible, dtype=bool)
    all_eligible = np.flatnonzero(eligible).astype(np.int64)
    if all_eligible.size == 0:
        raise RuntimeError("no eligible rows for SCConcept CITE-seq fine-tune")
    if int(max_rows) <= 0 or int(max_rows) >= all_eligible.size:
        return all_eligible

    rng = np.random.default_rng(int(seed))
    target = int(max_rows)
    dataset_arr = np.asarray(dataset_id)
    split_arr = np.asarray(split)
    groups = []
    for ds in np.unique(dataset_arr[all_eligible]):
        for sp in (0, 1):
            rows = np.flatnonzero(eligible & (dataset_arr == ds) & (split_arr == sp))
            if rows.size:
                groups.append(rows.astype(np.int64, copy=False))

    chosen = []
    remaining = target
    total = sum(int(x.size) for x in groups)
    for group_idx, rows in enumerate(groups):
        if group_idx == len(groups) - 1:
            take = remaining
        else:
            take = int(round(target * rows.size / max(1, total)))
            take = min(take, remaining)
        if take > 0:
            got = min(take, rows.size)
            chosen.append(rng.choice(rows, size=got, replace=False))
            remaining -= got
    out = np.sort(np.concatenate(chosen).astype(np.int64, copy=False)) if chosen else all_eligible[:target]
    if out.size < target:
        missing = target - out.size
        pool = np.setdiff1d(all_eligible, out, assume_unique=False)
        if pool.size:
            out = np.sort(np.concatenate([out, rng.choice(pool, size=min(missing, pool.size), replace=False)]))
    return out[:target].astype(np.int64, copy=False)

def _sample_log_int(rng, low: int, high: int) -> int:
    import numpy as np

    low = max(1, int(low))
    high = max(low, int(high))
    if low == high:
        return low
    value = int(round(float(np.exp(rng.uniform(np.log(low), np.log(high))))))
    return max(low, min(high, value))

def _resolve_panel_size(value: float | int, n_available: int) -> int:
    if float(value) <= 0:
        return 1
    if 0 < float(value) < 1:
        return max(1, int(math.ceil(float(value) * int(n_available))))
    return max(1, int(value))

def _sample_protein_panel_input(
    y,
    mask,
    *,
    rng,
    panel_size_min: int,
    panel_size_max: int,
    max_drop_rate: float,
    min_observed_per_cell: int,
):
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    n_cells, n_proteins = y.shape
    observed_cols = np.flatnonzero(mask.any(axis=0))
    if observed_cols.size == 0:
        y_view = np.zeros_like(y, dtype=np.float32)
        m_view = np.zeros_like(mask, dtype=bool)
        adt_input = np.concatenate([y_view, m_view.astype(np.float32)], axis=1)
        return adt_input.astype(np.float32), np.zeros(n_cells, dtype=bool), {"panel_size": 0, "available_proteins": 0}

    lo = min(max(1, int(panel_size_min)), observed_cols.size)
    hi = observed_cols.size if int(panel_size_max) <= 0 else min(int(panel_size_max), observed_cols.size)
    hi = max(lo, hi)
    panel_size = _sample_log_int(rng, lo, hi)
    panel_cols = np.sort(rng.choice(observed_cols, size=int(panel_size), replace=False).astype(np.int64))

    y_view = np.zeros_like(y, dtype=np.float32)
    m_view = np.zeros_like(mask, dtype=bool)
    m_panel = mask[:, panel_cols].copy()
    if float(max_drop_rate) > 0:
        drop_rate = float(rng.uniform(0.0, float(max_drop_rate)))
        m_panel &= rng.random(m_panel.shape) >= drop_rate
    y_panel = y[:, panel_cols].copy()
    y_panel[~m_panel] = 0.0
    y_view[:, panel_cols] = y_panel
    m_view[:, panel_cols] = m_panel
    valid = m_view.sum(axis=1) >= int(min_observed_per_cell)
    adt_input = np.concatenate([y_view, m_view.astype(np.float32)], axis=1)
    return adt_input.astype(np.float32), valid.astype(bool), {
        "panel_size": int(panel_size),
        "available_proteins": int(observed_cols.size),
        "valid_cells": int(valid.sum()),
        "observed_per_cell": _stats(m_view.sum(axis=1)),
    }

def _ensure_scconcept_model(model_cache_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    model_dir = model_cache_root / SC_MODEL_NAME
    required = [
        model_dir / "config.yaml",
        model_dir / "model.ckpt",
        model_dir / "gene_mappings" / "hsapiens.csv",
        model_dir / "pretrained_vocabulary" / "hsapiens.csv",
    ]
    if all(path.exists() for path in required):
        return model_dir
    model_cache_root.mkdir(parents=True, exist_ok=True)
    _log({"event": "download_scconcept_start", "repo": SC_REPO_ID, "revision": SC_REVISION})
    snapshot_download(
        repo_id=SC_REPO_ID,
        revision=SC_REVISION,
        local_dir=str(model_cache_root),
        allow_patterns=[SC_MODEL_GLOB],
        repo_type="model",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"scConcept snapshot is missing required files: {missing}")
    _log({"event": "download_scconcept_done", "model_dir": str(model_dir)})
    return model_dir

def _load_scconcept_for_training(device: str):
    import torch
    from concept import scConcept

    model_dir = _ensure_scconcept_model(MODEL_CACHE_ROOT)
    concept = scConcept(cache_dir=str(model_dir.parent))
    concept.load_config_and_model(
        config=model_dir / "config.yaml",
        model_path=model_dir / "model.ckpt",
        gene_mappings_path=model_dir / "gene_mappings",
        pretrained_vocabulary_path=model_dir / "pretrained_vocabulary",
    )
    model = concept.model.to(device)
    model.train()
    model.stage = "train"
    model.LOGGING_STEP = False
    model.world_size = 1
    model.val_loader_names = []
    model.set_active_species("hsapiens")
    model.use_learnable_embs_freq = 1.0
    torch.set_float32_matmul_precision("high")
    return concept, model, model_dir

def _gene_token_ids(concept, genes: list[str]):
    import numpy as np

    mapped = concept.map_gene_names_to_ids(species="hsapiens", gene_names=[str(g) for g in genes])
    mapped = np.asarray(mapped, dtype=object)
    token_ids = np.asarray(concept.tokenizer.encode(mapped, "hsapiens"), dtype=np.int64)
    valid = token_ids != int(concept.tokenizer.NOT_FOUND)
    return token_ids, valid

def _make_two_rna_panel_batches(
    dense,
    gene_token_ids,
    valid_gene_mask,
    *,
    pad_token: int,
    rng,
    panel_size_min: float,
    panel_size_max: int,
    max_tokens: int,
    min_genes_per_view: int,
):
    import numpy as np

    dense = np.asarray(dense, dtype=np.float32)
    dense = np.nan_to_num(dense, nan=0.0, posinf=65504.0, neginf=0.0)
    dense = np.clip(dense, 0.0, 65504.0)
    valid_gene_mask = np.asarray(valid_gene_mask, dtype=bool)
    observed_cols = np.flatnonzero(valid_gene_mask & (dense > 0).any(axis=0))
    if observed_cols.size == 0:
        return None

    lo = min(_resolve_panel_size(panel_size_min, observed_cols.size), observed_cols.size)
    hi = observed_cols.size if int(panel_size_max) <= 0 else min(int(panel_size_max), observed_cols.size)
    hi = max(lo, hi)
    size_1 = _sample_log_int(rng, lo, hi)
    cols_1 = np.sort(rng.choice(observed_cols, size=size_1, replace=False).astype(np.int64))

    remaining = np.setdiff1d(observed_cols, cols_1, assume_unique=False)
    if remaining.size >= lo:
        hi_2 = remaining.size if int(panel_size_max) <= 0 else min(int(panel_size_max), remaining.size)
        hi_2 = max(lo, hi_2)
        size_2 = _sample_log_int(rng, lo, hi_2)
        cols_2 = np.sort(rng.choice(remaining, size=size_2, replace=False).astype(np.int64))
        overlap = False
    else:
        size_2 = _sample_log_int(rng, lo, hi)
        cols_2 = np.sort(rng.choice(observed_cols, size=size_2, replace=False).astype(np.int64))
        overlap = bool(np.intersect1d(cols_1, cols_2).size)

    def panel_to_tensors(panel_cols):
        rows_tokens: list[np.ndarray] = []
        rows_values: list[np.ndarray] = []
        lengths: list[int] = []
        for row_values in dense[:, panel_cols]:
            nz = row_values > 0
            vals = row_values[nz]
            cols = panel_cols[nz]
            if vals.size:
                order = np.argsort(-vals, kind="stable")
                cols = cols[order][: int(max_tokens)]
                vals = vals[order][: int(max_tokens)]
                tokens = gene_token_ids[cols].astype(np.int64, copy=False)
                values = vals.astype(np.float32, copy=False)
            else:
                tokens = np.empty(0, dtype=np.int64)
                values = np.empty(0, dtype=np.float32)
            rows_tokens.append(tokens)
            rows_values.append(values)
            lengths.append(int(tokens.size))
        max_len = max(1, min(int(max_tokens), max(lengths) if lengths else 1))
        token_mat = np.full((dense.shape[0], max_len), int(pad_token), dtype=np.int64)
        value_mat = np.zeros((dense.shape[0], max_len), dtype=np.float32)
        for i, (tokens, values) in enumerate(zip(rows_tokens, rows_values)):
            take = min(max_len, int(tokens.size))
            if take:
                token_mat[i, :take] = tokens[:take]
                value_mat[i, :take] = values[:take]
            lengths[i] = take
        valid = np.asarray(lengths, dtype=np.int32) >= int(min_genes_per_view)
        return token_mat, value_mat, lengths, valid

    tokens_1, values_1, seq_1, valid_1 = panel_to_tensors(cols_1)
    tokens_2, values_2, seq_2, valid_2 = panel_to_tensors(cols_2)
    return {
        "tokens_1": tokens_1,
        "values_1": values_1,
        "seq_1": seq_1,
        "valid_1": valid_1,
        "tokens_2": tokens_2,
        "values_2": values_2,
        "seq_2": seq_2,
        "valid_2": valid_2,
        "meta": {
            "available_genes": int(observed_cols.size),
            "panel_size_1": int(cols_1.size),
            "panel_size_2": int(cols_2.size),
            "panel_overlap": overlap,
            "seq_len_1": _stats(seq_1),
            "seq_len_2": _stats(seq_2),
        },
    }

def _make_dataset_class():
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class CiteSeqFineTuneDataset(Dataset):
        def __init__(self, rows, protein, mask, protein_mean, protein_std, dataset_id) -> None:
            self.rows = np.asarray(rows, dtype=np.int64)
            self.protein = protein
            self.mask = mask
            self.dataset_id = dataset_id
            self.protein_mean = np.asarray(protein_mean, dtype=np.float32)
            self.protein_std = np.asarray(protein_std, dtype=np.float32)

        def __len__(self) -> int:
            return int(self.rows.size)

        def __getitem__(self, idx: int):
            row = int(self.rows[int(idx)])
            y = np.asarray(self.protein[row], dtype=np.float32).copy()
            m = np.asarray(self.mask[row], dtype=np.bool_).copy()
            y = np.nan_to_num((y - self.protein_mean) / self.protein_std, nan=0.0, posinf=0.0, neginf=0.0)
            y[~m] = 0.0
            return {
                "row": row,
                "protein": torch.from_numpy(y),
                "protein_mask": torch.from_numpy(m),
                "dataset_id": int(self.dataset_id[row]),
            }

    return CiteSeqFineTuneDataset

def _make_grouped_batch_sampler_class():
    import numpy as np
    from torch.utils.data import Sampler

    class WithinDatasetBatchSampler(Sampler):
        def __init__(self, dataset_ids, batch_size: int, *, seed: int, shuffle: bool, drop_last: bool) -> None:
            self.dataset_ids = np.asarray(dataset_ids)
            self.batch_size = int(batch_size)
            self.seed = int(seed)
            self.shuffle = bool(shuffle)
            self.drop_last = bool(drop_last)
            self.epoch = 0
            self.group_positions = {
                int(ds): np.flatnonzero(self.dataset_ids == ds).astype(np.int64, copy=False)
                for ds in np.unique(self.dataset_ids)
            }

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def __len__(self) -> int:
            total = 0
            for positions in self.group_positions.values():
                if self.drop_last:
                    total += int(positions.size) // self.batch_size
                else:
                    total += int(math.ceil(int(positions.size) / self.batch_size))
            return total

        def __iter__(self):
            rng = np.random.default_rng(self.seed + self.epoch * 1009)
            batches: list[np.ndarray] = []
            for ds in sorted(self.group_positions):
                positions = self.group_positions[ds]
                positions = rng.permutation(positions) if self.shuffle else positions.copy()
                for start in range(0, positions.size, self.batch_size):
                    batch = positions[start : start + self.batch_size]
                    if self.drop_last and batch.size < self.batch_size:
                        continue
                    if batch.size:
                        batches.append(batch)
            if self.shuffle:
                rng.shuffle(batches)
            for batch in batches:
                yield batch.tolist()

    return WithinDatasetBatchSampler

def _make_adt_encoder(input_dim: int, embedding_dim: int, hidden_dim: int, dropout: float):
    import torch.nn as nn

    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, embedding_dim),
        nn.LayerNorm(embedding_dim),
    )

def _encode_scconcept_view(model, tokens, values, seq_lengths, *, device: str, stage: str):
    import torch
    import torch.nn.functional as F

    model.stage = stage
    model.LOGGING_STEP = False
    model.set_active_species("hsapiens")
    batch = {
        "tokens": torch.as_tensor(tokens, device=device, dtype=torch.long),
        "values": torch.as_tensor(values, device=device, dtype=torch.float32),
        "seq_lengths": [int(x) for x in seq_lengths],
    }
    batch = model.add_cls_token(batch)
    _pred, _embs, cell_embs = model(batch["tokens"], batch["values"], seq_lengths=batch["seq_lengths"])
    if getattr(model, "projection_dim", None):
        cell_embs = model.projection(cell_embs)
    return F.normalize(cell_embs.float(), p=2, dim=1)

def _clip_loss(z_left, z_right, logit_scale, valid_mask):
    import torch
    import torch.nn.functional as F

    valid_mask = valid_mask.bool()
    idx = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if int(idx.numel()) < 2:
        zero = (z_left.sum() + z_right.sum()) * 0.0
        return zero, {"top1": float("nan"), "top5": float("nan"), "n": int(idx.numel())}
    z_left = z_left[idx]
    z_right = z_right[idx]
    logits = logit_scale * (z_left @ z_right.T)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    with torch.no_grad():
        top = logits.argsort(dim=1, descending=True)
        top1 = (top[:, 0] == labels).float().mean()
        top5 = (
            (top[:, : min(5, top.shape[1])] == labels[:, None])
            .any(dim=1)
            .float()
            .mean()
        )
    return loss, {
        "top1": float(top1.detach().cpu().item()),
        "top5": float(top5.detach().cpu().item()),
        "n": int(idx.numel()),
    }

def _run_epoch(
    *,
    model,
    adt_encoder,
    loader,
    sampler,
    rna,
    gene_token_ids,
    valid_gene_mask,
    optimizer,
    args: dict[str, Any],
    rng,
    device: str,
    epoch: int,
    training: bool,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    if training:
        model.train()
        adt_encoder.train()
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
    else:
        model.eval()
        adt_encoder.eval()

    records: list[dict[str, Any]] = []
    skipped = 0
    max_batches = int(args["max_train_batches"] if training else args["max_val_batches"])
    stage = "train" if training else "val"

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        rows = np.asarray(batch["row"], dtype=np.int64)
        dense = np.asarray(rna[rows], dtype=np.float32)
        views = _make_two_rna_panel_batches(
            dense,
            gene_token_ids,
            valid_gene_mask,
            pad_token=int(args["pad_token"]),
            rng=rng,
            panel_size_min=float(args["rna_panel_size_min"]),
            panel_size_max=int(args["rna_panel_size_max"]),
            max_tokens=int(args["rna_max_tokens"]),
            min_genes_per_view=int(args["min_genes_per_sampled_view"]),
        )
        if views is None:
            skipped += 1
            continue

        y = batch["protein"].numpy()
        protein_mask = batch["protein_mask"].numpy()
        adt_input, protein_valid, protein_meta = _sample_protein_panel_input(
            y,
            protein_mask,
            rng=rng,
            panel_size_min=int(args["protein_panel_size_min"]),
            panel_size_max=int(args["protein_panel_size_max"]),
            max_drop_rate=float(args["protein_max_drop_rate"]),
            min_observed_per_cell=int(args["min_proteins_per_sampled_view"]),
        )
        valid_np = views["valid_1"] & views["valid_2"] & protein_valid
        if int(valid_np.sum()) < int(args["min_valid_pairs_per_batch"]):
            skipped += 1
            continue

        valid = torch.from_numpy(valid_np).to(device=device, dtype=torch.bool)
        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = str(device).startswith("cuda")
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                z_1 = _encode_scconcept_view(
                    model,
                    views["tokens_1"],
                    views["values_1"],
                    views["seq_1"],
                    device=device,
                    stage=stage,
                )
                z_2 = _encode_scconcept_view(
                    model,
                    views["tokens_2"],
                    views["values_2"],
                    views["seq_2"],
                    device=device,
                    stage=stage,
                )
                adt_tensor = torch.from_numpy(adt_input).to(device=device, dtype=torch.float32)
                z_adt = F.normalize(adt_encoder(adt_tensor).float(), p=2, dim=1)
                logit_scale = model.logit_scale.exp().clamp(max=100.0)
                rna_loss, rna_metrics = _clip_loss(z_1, z_2, logit_scale, valid)
                adt_1_loss, adt_1_metrics = _clip_loss(z_1, z_adt, logit_scale, valid)
                adt_2_loss, adt_2_metrics = _clip_loss(z_2, z_adt, logit_scale, valid)
                adt_loss = 0.5 * (adt_1_loss + adt_2_loss)
                loss = (
                    float(args["rna_contrastive_weight"]) * rna_loss
                    + float(args["protein_contrastive_weight"]) * adt_loss
                )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(model.parameters()) + list(adt_encoder.parameters()) if p.requires_grad],
                float(args["grad_clip"]),
            )
            optimizer.step()

        with torch.no_grad():
            panel_adt_cos = (z_1 * z_adt).sum(dim=-1)[valid].detach().cpu().float().numpy()
            rna_pair_cos = (z_1 * z_2).sum(dim=-1)[valid].detach().cpu().float().numpy()
        record = {
            "loss": float(loss.detach().cpu().item() if hasattr(loss, "detach") else loss),
            "rna_loss": float(rna_loss.detach().cpu().item()),
            "adt_loss": float(adt_loss.detach().cpu().item()),
            "rna_top1": rna_metrics["top1"],
            "rna_top5": rna_metrics["top5"],
            "adt_top1": float(np.nanmean([adt_1_metrics["top1"], adt_2_metrics["top1"]])),
            "adt_top5": float(np.nanmean([adt_1_metrics["top5"], adt_2_metrics["top5"]])),
            "valid_pairs": int(valid_np.sum()),
            "rna_panel_size_1": int(views["meta"]["panel_size_1"]),
            "rna_panel_size_2": int(views["meta"]["panel_size_2"]),
            "protein_panel_size": int(protein_meta["panel_size"]),
            "rna_pair_cosine": rna_pair_cos,
            "rna_adt_cosine": panel_adt_cos,
            "logit_scale": float(logit_scale.detach().cpu().item()),
        }
        records.append(record)

        if training and batch_idx % int(args["log_every_n_batches"]) == 0:
            _log(
                {
                    "event": "sciteconcept_batch",
                    "epoch": int(epoch),
                    "batch": int(batch_idx),
                    "loss": record["loss"],
                    "rna_loss": record["rna_loss"],
                    "adt_loss": record["adt_loss"],
                    "valid_pairs": record["valid_pairs"],
                    "rna_panel_size_1": record["rna_panel_size_1"],
                    "rna_panel_size_2": record["rna_panel_size_2"],
                    "protein_panel_size": record["protein_panel_size"],
                }
            )

    if not records:
        return {"batches": 0, "skipped_batches": int(skipped), "loss": float("nan")}

    def scalar_stats(key: str) -> dict[str, Any]:
        return _stats([r[key] for r in records])

    return {
        "batches": int(len(records)),
        "skipped_batches": int(skipped),
        "loss": float(np.mean([r["loss"] for r in records])),
        "rna_loss": float(np.mean([r["rna_loss"] for r in records])),
        "adt_loss": float(np.mean([r["adt_loss"] for r in records])),
        "rna_top1": float(np.nanmean([r["rna_top1"] for r in records])),
        "rna_top5": float(np.nanmean([r["rna_top5"] for r in records])),
        "adt_top1": float(np.nanmean([r["adt_top1"] for r in records])),
        "adt_top5": float(np.nanmean([r["adt_top5"] for r in records])),
        "valid_pairs": int(sum(r["valid_pairs"] for r in records)),
        "rna_panel_size_1": scalar_stats("rna_panel_size_1"),
        "rna_panel_size_2": scalar_stats("rna_panel_size_2"),
        "protein_panel_size": scalar_stats("protein_panel_size"),
        "rna_pair_cosine": _stats(np.concatenate([r["rna_pair_cosine"] for r in records])),
        "rna_adt_cosine": _stats(np.concatenate([r["rna_adt_cosine"] for r in records])),
        "logit_scale": float(np.mean([r["logit_scale"] for r in records])),
    }

def _checkpoint_payload(
    *,
    model,
    adt_encoder,
    args: dict[str, Any],
    model_dir: Path,
    meta: dict[str, Any],
    epoch: int,
    validation: dict[str, Any],
    train_history: list[dict[str, Any]],
):
    from omegaconf import OmegaConf

    scconcept_config = None
    try:
        hparams = getattr(model, "hparams", {})
        maybe_config = hparams.get("config") if hasattr(hparams, "get") else None
        if maybe_config is not None:
            scconcept_config = OmegaConf.to_container(maybe_config, resolve=True)
    except Exception as exc:
        scconcept_config = {"serialization_warning": repr(exc)}

    return {
        "model_class": "ScConceptCiteSeqModelFineTune",
        "training_recipe": TRAINING_RECIPE,
        "created_unix": time.time(),
        "epoch": int(epoch),
        "selection_metric": str(args["selection_metric"]),
        "selection_metric_value": float(validation["loss"]),
        "scconcept_state_dict": model.state_dict(),
        "adt_encoder_state_dict": adt_encoder.state_dict(),
        "config": dict(args),
        "scconcept_config": scconcept_config,
        "scconcept_identity": {
            "source": "scconcept",
            "repo_id": SC_REPO_ID,
            "revision": SC_REVISION,
            "model_name": SC_MODEL_NAME,
            "model_dir": str(model_dir),
            "embedding_dim": int(args["embedding_dim"]),
            "rna_input_space": "raw_counts_rank_encoding",
            "view": "two_sampled_rna_panels_plus_sampled_adt_panel",
            "pack_n_genes": int(meta["n_genes"]),
            "pack_n_proteins": int(meta["n_proteins"]),
        },
        "best_validation": validation,
        "train_history": train_history,
    }

def _is_better(metric: str, new_value: float, old_value: float | None) -> bool:
    if old_value is None:
        return True
    if metric == "val_loss":
        return new_value < old_value
    raise ValueError(f"unknown selection metric {metric!r}")

def _train_sciteconcept(args: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    start_time = time.time()
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", args)

    pack_dir = Path(args["pack_dir"])
    meta, rna, protein, mask, split, dataset_id = _load_pack(pack_dir)
    observed = _observed_counts(mask, chunk_rows=int(args["stats_chunk_rows"]))
    eligible = observed >= int(args["min_observed_proteins"])
    selected_rows = _select_rows(split, dataset_id, eligible, max_rows=int(args["max_rows"]), seed=int(args["seed"]))
    train_rows = selected_rows[np.asarray(split[selected_rows]) == 0]
    val_rows = selected_rows[np.asarray(split[selected_rows]) == 1]
    if train_rows.size == 0 or val_rows.size == 0:
        raise RuntimeError(f"bad split after selection: train={train_rows.size}, val={val_rows.size}")
    np.save(output_dir / "selected_rows.npy", selected_rows)

    norm = _protein_norm_stats(protein, mask, train_rows, chunk_rows=int(args["stats_chunk_rows"]))
    np.save(output_dir / "protein_norm_mean.npy", norm["mean"])
    np.save(output_dir / "protein_norm_std.npy", norm["std"])
    np.save(output_dir / "protein_norm_count.npy", norm["count"])

    device = str(args["device"])
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    concept, model, model_dir = _load_scconcept_for_training(device)
    gene_token_ids, valid_gene_mask = _gene_token_ids(concept, [str(g) for g in meta["genes"]])
    args["pad_token"] = int(concept.tokenizer.PAD_TOKEN)
    args["not_found_token"] = int(concept.tokenizer.NOT_FOUND)
    args["embedding_dim"] = int(model.dim_model if not getattr(model, "projection_dim", None) else model.projection_dim)

    DatasetClass = _make_dataset_class()
    SamplerClass = _make_grouped_batch_sampler_class()
    train_dataset = DatasetClass(train_rows, protein, mask, norm["mean"], norm["std"], dataset_id)
    val_dataset = DatasetClass(val_rows, protein, mask, norm["mean"], norm["std"], dataset_id)
    train_sampler = SamplerClass(
        dataset_id[train_rows],
        int(args["batch_size"]),
        seed=int(args["seed"]),
        shuffle=True,
        drop_last=True,
    )
    val_sampler = SamplerClass(
        dataset_id[val_rows],
        int(args["batch_size"]),
        seed=int(args["seed"]) + 17,
        shuffle=False,
        drop_last=False,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0, pin_memory=False)

    adt_encoder = _make_adt_encoder(
        input_dim=int(meta["n_proteins"]) * 2,
        embedding_dim=int(args["embedding_dim"]),
        hidden_dim=int(args["adt_hidden_dim"]),
        dropout=float(args["dropout"]),
    ).to(device)

    resume_checkpoint = str(args.get("resume_checkpoint", "") or "").strip()
    resumed_from: dict[str, Any] | None = None
    resume_epoch_offset = 0
    resume_history: list[dict[str, Any]] = []
    resume_best_value: float | None = None
    resume_best_validation: dict[str, Any] | None = None
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["scconcept_state_dict"], strict=True)
        adt_encoder.load_state_dict(ckpt["adt_encoder_state_dict"], strict=True)
        resume_epoch_offset = int(ckpt.get("epoch", 0))
        resume_history = list(ckpt.get("train_history") or [])
        resume_best_validation = ckpt.get("best_validation") or None
        metric_value = ckpt.get("selection_metric_value")
        if metric_value is None and isinstance(resume_best_validation, dict):
            metric_value = resume_best_validation.get("loss")
        resume_best_value = float(metric_value) if metric_value is not None else None
        resumed_from = {
            "path": str(resume_path),
            "checkpoint_epoch": int(resume_epoch_offset),
            "checkpoint_metric": resume_best_value,
            "checkpoint_tag": str((ckpt.get("config") or {}).get("tag", "")),
        }
        _log({"event": "loaded_sciteconcept_resume", **resumed_from})

    params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": float(args["scconcept_lr"])},
        {"params": [p for p in adt_encoder.parameters() if p.requires_grad], "lr": float(args["adt_lr"])},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=float(args["weight_decay"]))

    trainable_scconcept = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_adt = sum(p.numel() for p in adt_encoder.parameters() if p.requires_grad)
    _log(
        {
            "event": "sciteconcept_start",
            "tag": args["tag"],
            "device": device,
            "n_cells": int(meta["n_cells"]),
            "n_genes": int(meta["n_genes"]),
            "n_proteins": int(meta["n_proteins"]),
            "selected_rows": int(selected_rows.size),
            "train_rows": int(train_rows.size),
            "val_rows": int(val_rows.size),
            "valid_genes_in_scconcept_vocab": int(np.sum(valid_gene_mask)),
            "trainable_scconcept_params": int(trainable_scconcept),
            "trainable_adt_params": int(trainable_adt),
            "scconcept_lr": float(args["scconcept_lr"]),
            "adt_lr": float(args["adt_lr"]),
            "resumed_from": resumed_from,
        }
    )

    rng = np.random.default_rng(int(args["seed"]) + int(resume_epoch_offset) * 100003)
    best_value: float | None = resume_best_value
    best_epoch = int(resume_epoch_offset) if resume_best_value is not None else 0
    best_validation: dict[str, Any] | None = resume_best_validation
    history: list[dict[str, Any]] = resume_history
    best_path = output_dir / "sciteconcept_best.pt"
    final_path = output_dir / "sciteconcept_final.pt"

    if resumed_from is not None and best_value is not None and best_validation is not None:
        torch.save(
            _checkpoint_payload(
                model=model,
                adt_encoder=adt_encoder,
                args=args,
                model_dir=model_dir,
                meta=meta,
                epoch=best_epoch,
                validation=best_validation,
                train_history=history,
            ),
            best_path,
        )
        _log(
            {
                "event": "saved_sciteconcept_resume_baseline",
                "path": str(best_path),
                "value": best_value,
                "epoch": best_epoch,
            }
        )

    for local_epoch in range(1, int(args["epochs"]) + 1):
        epoch = int(resume_epoch_offset) + int(local_epoch)
        epoch_start = time.time()
        train_metrics = _run_epoch(
            model=model,
            adt_encoder=adt_encoder,
            loader=train_loader,
            sampler=train_sampler,
            rna=rna,
            gene_token_ids=gene_token_ids,
            valid_gene_mask=valid_gene_mask,
            optimizer=optimizer,
            args=args,
            rng=rng,
            device=device,
            epoch=epoch,
            training=True,
        )
        val_metrics = _run_epoch(
            model=model,
            adt_encoder=adt_encoder,
            loader=val_loader,
            sampler=val_sampler,
            rna=rna,
            gene_token_ids=gene_token_ids,
            valid_gene_mask=valid_gene_mask,
            optimizer=optimizer,
            args=args,
            rng=rng,
            device=device,
            epoch=epoch,
            training=False,
        )
        epoch_payload = {
            "epoch": int(epoch),
            "seconds": round(time.time() - epoch_start, 3),
            "train": train_metrics,
            "validation": val_metrics,
            "logit_scale": float(model.logit_scale.detach().exp().clamp(max=100.0).cpu().item()),
        }
        history.append(epoch_payload)
        _log({"event": "sciteconcept_epoch", **epoch_payload})

        metric_value = float(val_metrics["loss"])
        if _is_better(str(args["selection_metric"]), metric_value, best_value):
            best_value = metric_value
            best_epoch = int(epoch)
            best_validation = val_metrics
            torch.save(
                _checkpoint_payload(
                    model=model,
                    adt_encoder=adt_encoder,
                    args=args,
                    model_dir=model_dir,
                    meta=meta,
                    epoch=epoch,
                    validation=val_metrics,
                    train_history=history,
                ),
                best_path,
            )
            _log({"event": "saved_sciteconcept_checkpoint", "path": str(best_path), "value": best_value})

    torch.save(
        _checkpoint_payload(
            model=model,
            adt_encoder=adt_encoder,
            args=args,
            model_dir=model_dir,
            meta=meta,
            epoch=int(resume_epoch_offset) + int(args["epochs"]),
            validation=best_validation or {},
            train_history=history,
        ),
        final_path,
    )
    report = {
        "event": "sciteconcept_training_complete",
        "run_name": str(args["tag"]),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "best_epoch": int(best_epoch),
        "best_value": float(best_value) if best_value is not None else None,
        "output_dir": str(output_dir),
        "resumed_from": resumed_from,
        "history": history,
        "artifacts": {
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "training_report": str(output_dir / "training_report.json"),
            "selected_rows": str(output_dir / "selected_rows.npy"),
            "protein_normalization_mean": str(output_dir / "protein_norm_mean.npy"),
            "protein_normalization_std": str(output_dir / "protein_norm_std.npy"),
        },
    }
    _write_json(output_dir / "training_report.json", report)
    _log(report)
    return report

def release_defaults() -> dict[str, Any]:
    """Hyperparameters used by the selected release trajectory."""
    return {
        "tag": "sciteconcept-v1",
        "device": "cuda",
        "max_rows": 131072,
        "min_observed_proteins": 4,
        "stats_chunk_rows": 65536,
        "rna_panel_size_min": 0.02,
        "rna_panel_size_max": 2048,
        "rna_max_tokens": 2048,
        "protein_panel_size_min": 8,
        "protein_panel_size_max": 512,
        "protein_max_drop_rate": 0.25,
        "min_genes_per_sampled_view": 8,
        "min_proteins_per_sampled_view": 2,
        "min_valid_pairs_per_batch": 16,
        "adt_hidden_dim": 2048,
        "dropout": 0.05,
        "scconcept_lr": 1.0e-5,
        "adt_lr": 1.0e-4,
        "weight_decay": 1.0e-4,
        "rna_contrastive_weight": 0.5,
        "protein_contrastive_weight": 1.0,
        "grad_clip": 1.0,
        "epochs": 4,
        "batch_size": 32,
        "max_train_batches": 0,
        "max_val_batches": 64,
        "log_every_n_batches": 25,
        "seed": 20260704,
        "selection_metric": "val_loss",
        "resume_checkpoint": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional JSON config; CLI values override it.")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    config = release_defaults()
    if cli.config:
        config.update(json.loads(cli.config.read_text(encoding="utf-8")))

    config.update({
        "pack_dir": str(cli.pack_dir),
        "output_dir": str(cli.output_dir),
    })
    overrides = {
        "tag": cli.run_name,
        "resume_checkpoint": str(cli.resume_checkpoint) if cli.resume_checkpoint else None,
        "epochs": cli.epochs,
        "max_rows": cli.max_rows,
        "batch_size": cli.batch_size,
        "device": cli.device,
        "max_train_batches": cli.max_train_batches,
        "max_val_batches": cli.max_val_batches,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    _train_sciteconcept(config)


if __name__ == "__main__":
    main()
