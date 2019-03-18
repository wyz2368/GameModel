import random
import numpy as np

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)