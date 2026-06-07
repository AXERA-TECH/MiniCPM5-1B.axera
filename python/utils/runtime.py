from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def default_hf_model(script_path: str) -> str:
    script_dir = Path(script_path).resolve().parent
    candidate = script_dir / "MiniCPM5-1B"
    return str(candidate)


def default_axmodel_path(script_path: str) -> str:
    script_dir = Path(script_path).resolve().parent
    return str(script_dir / "MiniCPM5-1B_axmodel")


def load_text_embeddings(axmodel_path: str, config) -> np.ndarray:
    npy_path = os.path.join(axmodel_path, "model.embed_tokens.weight.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)

    bf16_bin = os.path.join(axmodel_path, "model.embed_tokens.weight.bfloat16.bin")
    if os.path.exists(bf16_bin):
        return np.memmap(
            bf16_bin,
            mode="r",
            dtype=np.uint16,
        ).view(bfloat16).reshape(config.vocab_size, config.hidden_size)

    raise FileNotFoundError(
        "Cannot find model.embed_tokens.weight.npy or "
        "model.embed_tokens.weight.bfloat16.bin in axmodel_path"
    )
