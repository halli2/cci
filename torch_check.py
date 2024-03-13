import os

import torch

print(f"GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print(
        f"Mem:\n Allocated: {round(torch.cuda.memory_allocated(0) / 1024 **3, 1)}GB\n Cached: {round(torch.cuda.memory_reserved(0)/1024**3, 1)}GB"
    )
