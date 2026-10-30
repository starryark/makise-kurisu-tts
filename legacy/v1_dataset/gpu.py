"""GPU guard rails.

This box runs other hardware-accelerated work at the same time, so the labelling
stages must never take the whole card. Everything here is about staying a polite
tenant:

* hard-cap this process at ``CUDA_FRACTION`` (default 0.80) of total VRAM, so the
  allocator raises a normal Python OOM instead of pushing the driver into a state
  that takes the desktop down with it;
* small batches by default, with automatic bisection down to batch-1 on OOM;
* free the cache between batches so the resident set stays near the working set.

Override with the environment variables ``CUDA_FRACTION`` and ``VB_BATCH``.
"""
from __future__ import annotations

import gc
import os

import torch

FRACTION = float(os.environ.get("CUDA_FRACTION", "0.80"))
DEFAULT_BATCH = int(os.environ.get("VB_BATCH", "4"))

_configured = False


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure(verbose: bool = True) -> str:
    """Apply the VRAM cap. Safe to call more than once."""
    global _configured
    dev = device()
    if _configured or dev != "cuda":
        return dev
    torch.cuda.init()
    # CUDA_FRACTION >= 1 means "take the whole card" - do not install a cap at all.
    if FRACTION < 1.0:
        torch.cuda.set_per_process_memory_fraction(FRACTION, 0)
    # Fragmentation is what usually turns a tight cap into a hard failure.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if verbose:
        cap = (f"cap {FRACTION:.0%} of {total:.1f} GiB = {total * FRACTION:.1f} GiB"
               if FRACTION < 1.0 else f"UNCAPPED ({total:.1f} GiB)")
        print(f"[gpu] {torch.cuda.get_device_name(0)}  {cap}  batch {DEFAULT_BATCH}")
    _configured = True
    return dev


def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def used_gib() -> float:
    return torch.cuda.memory_reserved(0) / 2 ** 30 if torch.cuda.is_available() else 0.0


def run_batched(items: list, fn, batch: int | None = None, label: str = "",
                progress_every: int = 20):
    """Map ``fn(chunk) -> list`` over ``items``, halving the chunk on OOM.

    Returns results in input order. A chunk that still OOMs at size 1 raises, so a
    genuine bug is not silently swallowed.
    """
    batch = batch or DEFAULT_BATCH
    out: list = []
    i = 0
    n_done = 0
    while i < len(items):
        size = min(batch, len(items) - i)
        while True:
            try:
                out.extend(fn(items[i:i + size]))
                break
            except torch.OutOfMemoryError:
                free()
                if size == 1:
                    raise
                size = max(1, size // 2)
                print(f"[gpu] OOM -> retrying with batch {size}")
        i += size
        n_done += size
        if progress_every and (n_done // progress_every) != ((n_done - size) // progress_every):
            print(f"   {label} {i}/{len(items)}  (vram {used_gib():.1f} GiB)", flush=True)
            # Reclaiming after every batch throttles throughput for no benefit; the
            # allocator reuses its blocks. Do it occasionally, and always on OOM.
            if FRACTION < 1.0:
                free()
    return out
