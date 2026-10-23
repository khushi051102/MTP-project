"""Small helpers shared by every experiment."""
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed every random generator that can affect results.

    Python's `random`, NumPy and PyTorch each keep their OWN generator,
    so seeding one does not seed the others.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:          # torch not needed for the classical-ML phases
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
