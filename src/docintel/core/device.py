from __future__ import annotations

from typing import Literal

DeviceName = Literal["cpu", "cuda", "mps"]


def resolve_device(pref: str = "auto") -> DeviceName:
    """Pick cuda > mps > cpu. Torch is optional so Mac `dev` sync works without GPU wheels."""
    if pref != "auto":
        if pref not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"device must be auto|cpu|cuda|mps, got {pref!r}")
        return pref  # type: ignore[return-value]
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
