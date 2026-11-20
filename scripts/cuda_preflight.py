#!/usr/bin/env python3
"""Fail fast unless the remote runtime has a usable CUDA device."""

import sys

import torch


def main() -> None:
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_PREFLIGHT_FAILED: no CUDA device")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    sample = torch.randn((1024, 1024), device="cuda")
    torch.cuda.synchronize()
    _ = sample @ sample
    torch.cuda.synchronize()
    print("CUDA_PREFLIGHT_OK")


if __name__ == "__main__":
    main()
