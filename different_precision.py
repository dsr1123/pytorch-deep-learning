import os
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim=4096, hidden=8192, out_dim=4096, depth=6):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim)]
            d = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.float().norm(2).item() ** 2)
    return math.sqrt(total)


@torch.no_grad()
def maybe_reset_cuda_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(mode: str, device, steps=200, warmup=50, batch=64, in_dim=4096, dtype="bf16", lr=1e-3):
    """
    mode:
      - fp32: full float32
      - amp:  autocast mixed precision (fp16 or bf16), params remain fp32
    dtype (only for mode=amp):
      - fp16
      - bf16
    """
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = MLP(in_dim=in_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # synthetic data
    x = torch.randn(batch, in_dim, device=device)
    target = torch.randn(batch, in_dim, device=device)

    use_amp = (mode == "amp")
    if dtype == "fp16":
        amp_dtype = torch.float16
    elif dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        raise ValueError("dtype must be fp16 or bf16")

    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16 and device.type == "cuda"))

    times = []
    last_loss = None
    last_gn = None

    maybe_reset_cuda_stats(device)

    for i in range(warmup + steps):
        optim.zero_grad(set_to_none=True)

        t0 = time.perf_counter()

        if mode == "fp32":
            y = model(x.float())
            loss = F.mse_loss(y, target.float())
            loss.backward()
            optim.step()

        elif mode == "amp":
            # autocast works on CUDA; on CPU bf16 autocast may work depending on your build
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    y = model(x)
                    loss = F.mse_loss(y, target)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                # CPU fallback: run fp32 (so you can still execute the script)
                y = model(x.float())
                loss = F.mse_loss(y, target.float())
                loss.backward()
                optim.step()
        else:
            raise ValueError("mode must be fp32 or amp")

        sync(device)
        t1 = time.perf_counter()

        if i >= warmup:
            times.append((t1 - t0) * 1000.0)  # ms

        last_loss = float(loss.detach().float().item())
        last_gn = grad_norm(model)

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times)//2]
    p95 = sorted(times)[int(len(times)*0.95)]

    return {
        "mode": mode if mode == "fp32" else f"amp({dtype})",
        "avg_ms": avg,
        "p50_ms": p50,
        "p95_ms": p95,
        "last_loss": last_loss,
        "grad_norm": last_gn,
        "peak_mem_mb": peak_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--in_dim", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    r_fp32 = run("fp32", device, steps=args.steps, warmup=args.warmup, batch=args.batch, in_dim=args.in_dim)
    r_amp  = run("amp",  device, steps=args.steps, warmup=args.warmup, batch=args.batch, in_dim=args.in_dim, dtype=args.dtype)

    def fmt(r):
        mem = "-" if r["peak_mem_mb"] is None else f'{r["peak_mem_mb"]:.1f} MB'
        return (f'{r["mode"]:>10} | avg {r["avg_ms"]:.2f} ms | p50 {r["p50_ms"]:.2f} | p95 {r["p95_ms"]:.2f} '
                f'| loss {r["last_loss"]:.6g} | grad_norm {r["grad_norm"]:.3g} | peak_mem {mem}')

    print("\n=== Results (per step: forward+backward+optimizer) ===")
    print(fmt(r_fp32))
    print(fmt(r_amp))

    speedup = r_fp32["avg_ms"] / r_amp["avg_ms"]
    print(f"\nSpeedup: {speedup:.2f}x  (FP32 avg / AMP avg)")

    if device.type == "cuda":
        if r_fp32["peak_mem_mb"] and r_amp["peak_mem_mb"]:
            mem_ratio = r_amp["peak_mem_mb"] / r_fp32["peak_mem_mb"]
            print(f"Memory ratio: {mem_ratio:.2f}  (AMP / FP32)")

    print("\nTips:")
    print("- 如果你是 A100/H100/4090 等，建议优先试 bf16（更稳，通常也快）。")
    print("- 如果看到 AMP 比 FP32 还慢，通常是 batch 太小/模型太小/或 GPU 不支持/算子没走 Tensor Core。")


if __name__ == "__main__":
    
    main()