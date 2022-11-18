import numpy as np
from enum import Enum

celd_type = Enum('celd_type', 'unit empty')

def get_grid(height, width):
    return np.matrix([[celd_type.empty]*width]*height, copy=False)