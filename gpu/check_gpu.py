import sys
import torch, transformers
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("CUDA not available: need the cu128 PyTorch build and NVIDIA driver 570+"); sys.exit(1)
p = torch.cuda.get_device_properties(0)
print("GPU:", p.name, "| VRAM, GB:", round(p.total_memory / 2**30, 1), "| compute capability:", p.major, p.minor)
x = torch.randn(2048, 2048, device="cuda"); y = x @ x; torch.cuda.synchronize(); print("matmul on GPU: OK")
print("transformers", transformers.__version__)
try:
    import bitsandbytes as bnb; print("bitsandbytes", bnb.__version__)
except Exception as e:
    print("bitsandbytes: ERROR", e, "- 4-bit LLM loading will not work; report to Claude")
