from __future__ import annotations

from types import SimpleNamespace

from starVLA.training.trainer_utils.distributed import (
    configure_cuda_device_from_env,
    distributed_barrier,
)


class _Cuda:
    def __init__(self) -> None:
        self.selected: list[int] = []

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 8

    def set_device(self, device: int) -> None:
        self.selected.append(device)

    @staticmethod
    def current_device() -> int:
        return 3


class _Dist:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def get_backend() -> str:
        return "nccl"

    def barrier(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_configure_cuda_device_uses_local_rank(monkeypatch):
    cuda = _Cuda()
    monkeypatch.setenv("LOCAL_RANK", "11")
    monkeypatch.setattr(
        "starVLA.training.trainer_utils.distributed.torch",
        SimpleNamespace(cuda=cuda),
    )

    configure_cuda_device_from_env()

    assert cuda.selected == [3]


def test_distributed_barrier_passes_current_cuda_device_for_nccl(monkeypatch):
    dist = _Dist()
    monkeypatch.setattr(
        "starVLA.training.trainer_utils.distributed.dist",
        dist,
    )
    monkeypatch.setattr(
        "starVLA.training.trainer_utils.distributed.torch",
        SimpleNamespace(cuda=_Cuda()),
    )

    distributed_barrier()

    assert dist.calls == [{"device_ids": [3]}]
