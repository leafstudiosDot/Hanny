import time
import torch

print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("No CUDA GPU visible to PyTorch. Check `nvidia-smi` and that you installed the CUDA build of torch.")
    raise SystemExit(1)

props = torch.cuda.get_device_properties(0)
print("GPU:", props.name)
print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
print("Compute capability:", f"{props.major}.{props.minor}")

# Simple speed test: 2048x2048 matmul, CPU vs GPU (float32)
n = 2048
a = torch.randn(n, n)
b = torch.randn(n, n)

t0 = time.perf_counter()
for _ in range(5):
    a @ b
cpu_t = (time.perf_counter() - t0) / 5

a_gpu, b_gpu = a.cuda(), b.cuda()
a_gpu @ b_gpu  # warm-up
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    a_gpu @ b_gpu
torch.cuda.synchronize()
gpu_t = (time.perf_counter() - t0) / 20

print(f"CPU matmul: {cpu_t * 1000:.1f} ms")
print(f"GPU matmul: {gpu_t * 1000:.1f} ms  ({cpu_t / gpu_t:.0f}x faster)")
print("M0.1 passed.")