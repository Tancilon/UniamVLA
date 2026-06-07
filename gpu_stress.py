import torch
import time
import argparse


def get_dtype(name: str):
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be fp32, fp16, or bf16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=12288, help="Matrix size, e.g. 4096, 8192, 12288")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seconds", type=int, default=0, help="Run duration. 0 means run forever")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    device = "cuda"
    dtype = get_dtype(args.dtype)

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Matrix size: {args.size} x {args.size}")
    print(f"Dtype: {args.dtype}")
    print("Press Ctrl+C to stop")

    a = torch.randn((args.size, args.size), device=device, dtype=dtype)
    b = torch.randn((args.size, args.size), device=device, dtype=dtype)

    # 预热
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.time()
    i = 0

    try:
        while True:
            c = torch.matmul(a, b)
            torch.cuda.synchronize()

            i += 1
            if i % 10 == 0:
                used = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"iter={i}, memory_allocated={used:.2f} GB, memory_reserved={reserved:.2f} GB")

            if args.sleep > 0:
                time.sleep(args.sleep)

            if args.seconds > 0 and time.time() - start >= args.seconds:
                break

    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
