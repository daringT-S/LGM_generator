"""Use the same seeding calls and placement as the seed-3407 baseline."""
def set_distillation_seed(seed):
    import random
    import numpy as np
    import torch
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Distillation seed must be a uint32 integer')
    torch.manual_seed(seed)  # Also seeds all CUDA devices, as in the baseline.
    np.random.seed(seed)
    random.seed(seed)
