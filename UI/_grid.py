import numpy as np
from enum import Enum

celd_type = Enum('celd_type', 'unit empty')

def get_grid(height: int, width: int) -> np.matrix:
    """
    Crea una grilla a partir de las dimensiones que se pasan en el método

    height -> alto del grid, width -> ancho del grid

    return -> np.matrix
    """
    return np.matrix([[celd_type.empty]*width]*height, copy=False)