#!/usr/bin/env python3
"""Keep a CUDA GPU warm with controlled memory and compute load.

This script is intended for machines you own or have permission to use. It
allocates a configurable fraction of GPU memory, then runs repeated matrix
multiplications with a duty cycle that approximates the requested utilization.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch


BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class DTypeSpec:
    torch_dtype: torch.dtype
    bytes_per_element: int


def get_dtype(name: str) -> DTypeSpec:
    if name == "fp32":
        return DTypeSpec(torch.float32, 4)
    if name == "fp16":
        return DTypeSpec(torch.float16, 2)
    if name == "bf16":
        return DTypeSpec(torch.bfloat16, 2)
    raise ValueError("dtype must be fp32, fp16, or bf16")


def gib(num_bytes: int | float) -> float:
    return float(num_bytes) / BYTES_PER_GIB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate a target fraction of GPU memory and generate an approximate "
            "GPU utilization load."
        )
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to use, e.g. 0. This is required to avoid accidental GPU 0 use.",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.20,
        help="Fraction of total GPU memory to allocate. 0.20 means about 20%%.",
    )
    parser.add_argument(
        "--target-util",
        type=float,
        default=0.80,
        help="Approximate duty-cycle target. 0.80 means busy for about 80%% of each cycle.",
    )
    parser.add_argument(
        "--cycle-seconds",
        type=float,
        default=1.0,
        help="Duty-cycle window length in seconds.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=8192,
        help="Square matrix size for torch.mm. Increase if utilization is too low.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Compute dtype. fp16 usually saturates RTX-class GPUs best.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=0,
        help="Run duration. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=10.0,
        help="Seconds between progress reports.",
    )
    parser.add_argument(
        "--ballast-chunk-gib",
        type=float,
        default=1.0,
        help="Chunk size for memory ballast allocations.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.mem_fraction < 0.95:
        raise ValueError("--mem-fraction must be between 0 and 0.95")
    if not 0.0 < args.target_util <= 1.0:
        raise ValueError("--target-util must be between 0 and 1")
    if args.cycle_seconds <= 0:
        raise ValueError("--cycle-seconds must be positive")
    if args.matrix_size <= 0:
        raise ValueError("--matrix-size must be positive")
    if args.report_interval <= 0:
        raise ValueError("--report-interval must be positive")
    if args.ballast_chunk_gib <= 0:
        raise ValueError("--ballast-chunk-gib must be positive")


def allocate_ballast(target_bytes: int, chunk_bytes: int, device: torch.device) -> list[torch.Tensor]:
    ballast: list[torch.Tensor] = []
    remaining = max(0, target_bytes - torch.cuda.memory_allocated(device))

    while remaining > 0:
        this_chunk = min(remaining, chunk_bytes)
        ballast.append(torch.empty(this_chunk, dtype=torch.uint8, device=device))
        remaining -= this_chunk

    return ballast


def create_compute_tensors(
    matrix_size: int,
    dtype_spec: DTypeSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (matrix_size, matrix_size)
    a = torch.randn(shape, dtype=dtype_spec.torch_dtype, device=device)
    b = torch.randn(shape, dtype=dtype_spec.torch_dtype, device=device)
    out = torch.empty(shape, dtype=dtype_spec.torch_dtype, device=device)
    return a, b, out


def warmup(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, rounds: int = 8) -> None:
    for _ in range(rounds):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize(a.device)


def report(prefix: str, device: torch.device, iterations: int) -> None:
    allocated = gib(torch.cuda.memory_allocated(device))
    reserved = gib(torch.cuda.memory_reserved(device))
    print(
        f"{prefix} iter={iterations} "
        f"memory_allocated={allocated:.2f} GiB "
        f"memory_reserved={reserved:.2f} GiB",
        flush=True,
    )


def run_load(
    args: argparse.Namespace,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    device: torch.device,
) -> None:
    busy_seconds = args.cycle_seconds * args.target_util
    idle_seconds = args.cycle_seconds - busy_seconds
    start = time.monotonic()
    next_report = start + args.report_interval
    iterations = 0

    print("Press Ctrl+C to stop.", flush=True)

    while True:
        cycle_start = time.monotonic()
        busy_until = cycle_start + busy_seconds

        while time.monotonic() < busy_until:
            torch.mm(a, b, out=out)
            torch.cuda.synchronize(device)
            iterations += 1

        if idle_seconds > 0:
            time.sleep(idle_seconds)

        now = time.monotonic()
        if now >= next_report:
            elapsed = now - start
            report(f"elapsed={elapsed:.0f}s", device, iterations)
            next_report = now + args.report_interval

        if args.seconds > 0 and now - start >= args.seconds:
            break


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(f"--device must be in [0, {torch.cuda.device_count() - 1}]")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype_spec = get_dtype(args.dtype)

    props = torch.cuda.get_device_properties(device)
    total_bytes = int(props.total_memory)
    target_bytes = int(total_bytes * args.mem_fraction)
    chunk_bytes = int(args.ballast_chunk_gib * BYTES_PER_GIB)

    compute_tensor_bytes = 3 * args.matrix_size * args.matrix_size * dtype_spec.bytes_per_element
    if compute_tensor_bytes >= target_bytes:
        min_fraction = compute_tensor_bytes / total_bytes
        raise ValueError(
            f"matrix tensors need {gib(compute_tensor_bytes):.2f} GiB, which exceeds the "
            f"target allocation {gib(target_bytes):.2f} GiB. Lower --matrix-size or set "
            f"--mem-fraction above {min_fraction:.3f}."
        )

    print(f"Using GPU {args.device}: {props.name}", flush=True)
    print(f"Total memory: {gib(total_bytes):.2f} GiB", flush=True)
    print(f"Target memory: {gib(target_bytes):.2f} GiB ({args.mem_fraction:.0%})", flush=True)
    print(
        f"Target utilization: about {args.target_util:.0%} "
        f"({args.cycle_seconds:.2f}s cycle, {args.dtype}, matrix={args.matrix_size})",
        flush=True,
    )

    a, b, out = create_compute_tensors(args.matrix_size, dtype_spec, device)
    ballast = allocate_ballast(target_bytes, chunk_bytes, device)
    warmup(a, b, out)

    report("initial", device, 0)
    try:
        run_load(args, a, b, out, device)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    finally:
        del ballast, a, b, out
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()