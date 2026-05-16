"""Small distributed helpers shared by trainers and dataset setup."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _env_int(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None


def configure_cuda_device_from_env() -> int | None:
    """Bind this process to its local CUDA device when a launcher exposes one."""
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None

    device_count = cuda.device_count()
    if device_count <= 0:
        return None

    local_rank = _env_int(
        "LOCAL_RANK",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
    )
    if local_rank is None:
        return None

    device = local_rank % device_count
    cuda.set_device(device)
    return device


def _dist_is_ready() -> bool:
    is_available = getattr(dist, "is_available", None)
    is_initialized = getattr(dist, "is_initialized", None)
    return bool(
        callable(is_available)
        and callable(is_initialized)
        and is_available()
        and is_initialized()
    )


def _dist_backend() -> str | None:
    get_backend = getattr(dist, "get_backend", None)
    if not callable(get_backend):
        return None
    try:
        return str(get_backend()).lower()
    except Exception:
        return None


def all_ranks_true(value: bool, device=None) -> bool:
    """Return True only when every distributed rank reports True."""
    if not _dist_is_ready():
        return bool(value)

    flag = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def distributed_barrier() -> None:
    """Synchronize ranks, passing device_ids for NCCL barriers when possible."""
    if not _dist_is_ready():
        return

    cuda = getattr(torch, "cuda", None)
    if _dist_backend() == "nccl" and cuda is not None and cuda.is_available():
        try:
            dist.barrier(device_ids=[cuda.current_device()])
            return
        except TypeError:
            pass

    dist.barrier()
