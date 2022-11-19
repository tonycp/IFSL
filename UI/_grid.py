import numpy as np
from enum import Enum

celd_type = Enum('celd_type', 'unit empty')

def get_color(celd: celd_type) -> tuple[int, int, int]:
    """
    Dado un tipo de celda retorna su representante en color

    return -> tuple[int, int, int]
    """
    if celd is celd_type.empty:
        return (0, 0, 0)
    if celd is celd_type.unit:
        return (0, 0, 0)
    raise NotImplementedError("celd type indefinido")

def get_grid(height: int, width: int) -> np.matrix:
    """
    Crea una grilla a partir de las dimensiones que se pasan en el método

    height -> alto del grid, width -> ancho del grid

    return -> np.matrix
    """
    return np.matrix([[celd_type.empty]*width]*height, copy=False)