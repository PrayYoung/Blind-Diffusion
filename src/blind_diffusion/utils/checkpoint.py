import os
import torch
from typing import Dict, Any


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
