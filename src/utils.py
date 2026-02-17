"""
Utility functions for reproducibility, file I/O, and directory management.
"""

import os
import json
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: dict, path: str) -> None:
    """Save dictionary as formatted JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Load JSON file and return as dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
