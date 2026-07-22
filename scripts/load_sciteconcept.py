"""Load the released epoch-94 sCITEconcept checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file


REPO_ID = "alunalabs/sCITEconcept"
BASE_REPO_ID = "theislab/scConcept"
BASE_MODEL_NAME = "corpus230M[human]-model170M"
BASE_REVISION = "10ed3ec8f35249247c33e1835e381a4c935ee26f"
BASE_MODEL_GLOB = "corpus230M[[]human[]]-model170M/**"
WEIGHTS_FILENAME = "sCITEconcept.safetensors"
METADATA_FILENAME = "checkpoint_metadata.json"


def _base_model_dir(cache_root: Path) -> Path:
    model_dir = cache_root / BASE_MODEL_NAME
    required = (
        model_dir / "config.yaml",
        model_dir / "model.ckpt",
        model_dir / "gene_mappings" / "hsapiens.csv",
        model_dir / "pretrained_vocabulary" / "hsapiens.csv",
    )
    if not all(path.exists() for path in required):
        snapshot_download(
            repo_id=BASE_REPO_ID,
            revision=BASE_REVISION,
            local_dir=cache_root,
            allow_patterns=[BASE_MODEL_GLOB],
            repo_type="model",
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"base scConcept snapshot is missing required files: {missing}")
    return model_dir


def load_sciteconcept(
    cache_dir: str | Path | None = None,
    *,
    token: str | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Return a configured ``concept.scConcept`` object and release metadata."""
    from concept import scConcept

    cache_root = Path(cache_dir or Path.home() / ".cache" / "sciteconcept")
    model_dir = _base_model_dir(cache_root / "theislab_scConcept")
    weights_path = hf_hub_download(REPO_ID, WEIGHTS_FILENAME, token=token)
    metadata_path = hf_hub_download(REPO_ID, METADATA_FILENAME, token=token)

    concept = scConcept(cache_dir=str(model_dir.parent))
    concept.load_config_and_model(
        config=model_dir / "config.yaml",
        model_path=model_dir / "model.ckpt",
        gene_mappings_path=model_dir / "gene_mappings",
        pretrained_vocabulary_path=model_dir / "pretrained_vocabulary",
    )
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata.get("model_class") != "ScConceptCiteSeqModelFineTune":
        raise RuntimeError(f"unexpected model_class: {metadata.get('model_class')!r}")
    concept.model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
    concept.model.to(map_location)
    concept.model.eval()
    concept.model.stage = "val"
    concept.model.LOGGING_STEP = False
    concept.model.world_size = 1
    concept.model.val_loader_names = []
    concept.model.set_active_species("hsapiens")
    concept.model.use_learnable_embs_freq = 1.0
    return concept, metadata
