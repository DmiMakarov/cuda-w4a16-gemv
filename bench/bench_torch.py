"""PyTorch baselines: dequantize the w4 matrix, then matmul; and matmul on ready fp16 weights.

Usage: python bench/bench_torch.py [rows] [cols] [G] [iterations]
Same CSV columns as bench_gemv (without bandwidth percentages). Codes are packed two per byte.
"""
import statistics
import sys
import time

import torch

rows, cols, G, iters = (int(v) for v in (sys.argv[1:] + ["16384", "16384", "32", "50"][len(sys.argv) - 1:]))
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("warning: no CUDA, timings are meaningless", file=sys.stderr)

groups = cols // G
gen = torch.Generator(device=device).manual_seed(42)
packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device=device, generator=gen)
scale = torch.full((rows, groups), 0.002, dtype=torch.float16, device=device)
zero = torch.full((rows, groups), 7, dtype=torch.uint8, device=device)
x = torch.randn(cols, device=device, generator=gen).half()


def dequantize():
    q = torch.stack((packed & 15, packed >> 4), dim=-1).view(rows, groups, G)
    w = (q.half() - zero.unsqueeze(-1).half()) * scale.unsqueeze(-1)
    return w.view(rows, cols)


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def measure(fn):
    for _ in range(5):
        fn()
    sync()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    return statistics.median(times), times[0], times[len(times) * 9 // 10]


def report(name, stats, mbytes):
    print(f"{name},{stats[0]:.2f},{stats[1]:.2f},{stats[2]:.2f},{mbytes:.2f}")


with torch.inference_mode():
    print("name,median_us,min_us,p90_us,matrix_MB")
    w_fp16 = dequantize()
    report("torch_dequant_matmul", measure(lambda: dequantize() @ x), packed.numel() / 1e6 + rows * groups * 3 / 1e6)
    report("torch_fp16_matmul", measure(lambda: w_fp16 @ x), w_fp16.numel() * 2 / 1e6)
