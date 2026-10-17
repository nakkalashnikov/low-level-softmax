import torch
import time

device = torch.device('mps') 
x = torch.randn(10000000, dtype=torch.float32, device=device)

start_time = time.perf_counter()
out_official = torch.nn.functional.softmax(x, dim=0)
official_time = time.perf_counter() - start_time

print(f"PyTorch Official Time: {official_time * 1000:.2f} ms")

start_time = time.perf_counter()

max_val = torch.max(x)

safe_x = x - max_val
exps = torch.exp(safe_x)

sum_exps = torch.sum(exps)

out_manual = exps / sum_exps
manual_time = time.perf_counter() - start_time

print(f"PyTorch Manual Step-by-Step Time: {manual_time * 1000:.2f} ms")

max_diff = torch.max(torch.abs(out_official - out_manual))
print(f"Max difference error: {max_diff.item()}")