import numpy as np
import torch
import random


def generate_random_seeds(seed, nums):
    random.seed(seed)
    return [random.randint(1, 999999999) for _ in range(nums)]


def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True