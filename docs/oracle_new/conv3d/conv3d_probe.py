"""Conv3d degenerate-dispatch probe: timing matrix + numerics consistency.

Usage: <env_python> conv3d_probe.py <out_json_path>
Run once per conda env on the SAME node. Covers:
  - Conv3d vs F.linear timing at N in {2040, 11360} for bf16/fp16/fp32
  - bf16 numerics delta (Conv3d vs Linear-equivalent) on GPU
  - fp64 CPU exactness reference (expect ~1e-13)
"""
import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = sys.argv[1]
torch.manual_seed(0)
O = 1024
KERNEL = (2, 16, 16)

res = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "benchmarks": [],
    "numerics": {},
}


def bench(fn, reps):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000.0


if torch.cuda.is_available():
    for n, reps in [(2040, 3), (11360, 1)]:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            conv = nn.Conv3d(3, O, kernel_size=KERNEL, stride=KERNEL, bias=True).cuda().to(dt)
            x = torch.randn(n, 3, *KERNEL, device="cuda", dtype=dt)
            xf = x.reshape(n, -1)
            w = conv.weight.view(O, -1)
            t_conv = bench(lambda: conv(x), reps)
            t_lin = bench(lambda: F.linear(xf, w, conv.bias), reps)
            res["benchmarks"].append({
                "N": n, "dtype": str(dt).split(".")[-1],
                "conv3d_ms": round(t_conv, 3), "linear_ms": round(t_lin, 3),
                "slowdown_x": round(t_conv / t_lin, 1),
            })
            del conv, x, xf, w
            torch.cuda.empty_cache()

# ── numerics ──────────────────────────────────────────────────────────────
n = 2040
conv = nn.Conv3d(3, O, kernel_size=KERNEL, stride=KERNEL, bias=True)
x64 = torch.randn(n, 3, *KERNEL, dtype=torch.float64)
c64 = conv.double()
y_conv = F.conv3d(x64, c64.weight, c64.bias, stride=KERNEL).reshape(n, -1)
y_lin = F.linear(x64.reshape(n, -1), c64.weight.view(O, -1), c64.bias)
res["numerics"]["fp64_cpu_max_abs"] = float((y_conv - y_lin).abs().max())

if torch.cuda.is_available():
    cb = conv.bfloat16().cuda()
    xb = x64.bfloat16().cuda()
    with torch.no_grad():
        yc = cb(xb).float().reshape(n, -1)
        yl = F.linear(xb.reshape(n, -1), cb.weight.view(O, -1), cb.bias).float()
    d = (yc - yl).abs()
    res["numerics"]["bf16_gpu_max_abs"] = float(d.max())
    res["numerics"]["bf16_gpu_mean_abs"] = float(d.mean())
    res["numerics"]["bf16_gpu_out_rms"] = float(yc.square().mean().sqrt())

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
