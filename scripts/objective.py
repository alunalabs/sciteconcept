"""The exact three-view contrastive objective used to train sCITEconcept."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _clip_pair(left: Tensor, right: Tensor, logit_scale: Tensor) -> Tensor:
    """Symmetric in-batch cross-entropy for one aligned pair of views."""
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left and right must be equal-shaped [batch, embedding] tensors")
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * left @ right.T
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def contrastive_objective(
    rna_a: Tensor,
    rna_b: Tensor,
    adt: Tensor,
    logit_scale: Tensor,
    *,
    rna_weight: float = 0.5,
    protein_weight: float = 1.0,
) -> Tensor:
    """Compute the sCITEconcept RNA/RNA plus RNA/protein training loss.

    This is the readable form of the production trainer's masked implementation:

    ``0.5 * CLIP(A, B) + 1.0 * 0.5 * (CLIP(A, ADT) + CLIP(B, ADT))``
    """
    rna_loss = _clip_pair(rna_a, rna_b, logit_scale)
    protein_loss = 0.5 * (
        _clip_pair(rna_a, adt, logit_scale) + _clip_pair(rna_b, adt, logit_scale)
    )
    return float(rna_weight) * rna_loss + float(protein_weight) * protein_loss

