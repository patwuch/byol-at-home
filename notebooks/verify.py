# verify.py — run this before anything else
import torch
import numpy as np
from PIL import Image
import torchvision

print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.version.cuda}")
print(f"GPU:      {torch.cuda.get_device_name(0)}")
print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Verify a tensor actually moves to GPU and back
x = torch.randn(4, 3, 96, 96).cuda()
print(f"Tensor on GPU: {x.device}")
print(f"CUDA working:  OK")