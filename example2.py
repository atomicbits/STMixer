import torch
import numpy as np
import random

def set_seed(seed, rank, world_size):
    rng = random.Random(seed)
    seed_per_rank = [rng.randint(0, 2**32-1) for _ in range(world_size)]
    print(seed_per_rank)
    cur_seed = seed_per_rank[rank]
    random.seed(cur_seed)
    torch.manual_seed(cur_seed)
    torch.cuda.manual_seed(cur_seed)
    np.random.seed(cur_seed)
    print(cur_seed, "\n")

# Example usage:
seed = 42
rank = 0  # This would be different for each process
world_size = 8  # Total number of processes
set_seed(seed, rank, world_size)
