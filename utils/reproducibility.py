"""
Utilities for ensuring reproducibility in training.
"""
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set the random seed for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed to use
        deterministic: Whether to use deterministic algorithms in PyTorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Enable cudnn auto-tuner for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True