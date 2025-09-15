import os
import random
from typing import Callable

import numpy as np
import torch


def fix_random_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch. Optionally enable fully-deterministic
    execution where supported. Should be called at the very start of each process.
    """
    # Python / NumPy / PyTorch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS determinism (must be set early in process)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # cuDNN knobs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enforce deterministic algorithms where available
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch versions may not support this API
            pass


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """
    Factory for DataLoader's worker_init_fn to ensure each worker has a stable RNG state
    derived from torch.initial_seed(). Also re-sets PYTHONHASHSEED inside the worker.
    """

    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random as _random  # local import to ensure correct module seeding

        _random.seed(worker_seed)
        os.environ["PYTHONHASHSEED"] = str(base_seed)

    return _seed_worker


def make_generator(seed: int, rank: int = 0) -> torch.Generator:
    """Create a torch.Generator seeded for reproducible DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(int(seed) + int(rank))
    return g


