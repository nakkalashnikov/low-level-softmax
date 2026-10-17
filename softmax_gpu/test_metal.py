import torch
import softmax_metal_backend
import time

N_rows = 10000
N_cols = 1024
x = torch.randn(N_rows, N_cols, device='mps')

_ = softmax_metal_backend.forward(x)
torch.mps.synchronize()

start = time.perf_counter()
y_custom = softmax_metal_backend.forward(x)
torch.mps.synchronize() 
custom_time = (time.perf_counter() - start) * 1000

start = time.perf_counter()
y_official = torch.nn.functional.softmax(x, dim=1)
torch.mps.synchronize()
official_time = (time.perf_counter() - start) * 1000

print(f"Custom Metal Time:   {custom_time:.4f} ms")
print(f"Official PyTorch:    {official_time:.4f} ms")
print(f"Speedup:             {official_time / custom_time:.2f}x")

diff = torch.abs(y_custom.to('mps') - y_official).max()
print(f"Max difference:      {diff.item()}")