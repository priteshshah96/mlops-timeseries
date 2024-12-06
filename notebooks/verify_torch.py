"""PyTorch verification script."""
from typing import Optional

import torch


def check_torch_setup() -> None:
    """Verify PyTorch and CUDA setup."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    device_name: Optional[str] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    print(f"GPU Device: {device_name}")


if __name__ == "__main__":
    check_torch_setup()
