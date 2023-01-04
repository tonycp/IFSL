from cmath import inf
from enum import Enum


I_DIR = [-1,-1,0,1,1,1,0,-1]
J_DIR = [0,1,1,1,0,-1,-1,-1]

DIRECTIONS = Enum('DIRECTIONS','N NE E SE S SW W NW')

STATES = Enum('STATES','Moving Swaping Attacking Stand')

def direction_to_int(direction):
    if direction == DIRECTIONS.N:
        return 0
    elif direction == DIRECTIONS.NE:
        return 1
    elif direction == DIRECTIONS.E:
        return 2
    elif direction == DIRECTIONS.SE:
        return 3
    elif direction == DIRECTIONS.S:
        return 4
    elif direction == DIRECTIONS.SW:
        return 5
    elif direction == DIRECTIONS.W:
        return 6
    elif direction == DIRECTIONS.NW:
        return 7

def direction_to_tuple(direction):
    if direction == DIRECTIONS.N:
        return (-1,0)
    elif direction == DIRECTIONS.NE:
        return (-1,1)
    elif direction == DIRECTIONS.E:
        return (0,1)
    elif direction == DIRECTIONS.SE:
        return (1,1)
    elif direction == DIRECTIONS.S:
        return (1,0)
    elif direction == DIRECTIONS.SW:
        return (1,-1)
    elif direction == DIRECTIONS.W:
        return (0,-1)
    elif direction == DIRECTIONS.NW:
        return (-1,-1)

dir_tuple = {
    (-1, 0): DIRECTIONS.N,
    (-1, 1): DIRECTIONS.NE,
    (0, 1): DIRECTIONS.E,
    (1, 1): DIRECTIONS.SE,
    (1, 0): DIRECTIONS.S,
    (1, -1): DIRECTIONS.SW,
    (0, -1): DIRECTIONS.W,
    (-1, -1): DIRECTIONS.NW
}

class Singleton(type):
    """
    clase usada para que todas las intancias hagan referencia al mismo objeto

    __instance -> unica instancia de la clase
    """
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        type.__init__(cls, *args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = type.__call__(cls, *args,**kwargs)
        return cls.__instance


def int_to_direction(direction):
    if direction == 0:
        return DIRECTIONS.N
    elif direction == 1:
        return DIRECTIONS.NE
    elif direction == 2:
        return DIRECTIONS.E
    elif direction == 3:
        return DIRECTIONS.SE
    elif direction == 4:
        return DIRECTIONS.S
    elif direction == 5:
        return DIRECTIONS.SW
    elif direction == 6:
        return DIRECTIONS.W
    elif direction == 7:
        return DIRECTIONS.NW
    
def validMove(x, y, max_x, max_y, min_x = 0, min_y = 0): 
    return  x >= min_x and x < max_x and y >= min_y and y < max_y

def norma2(n1, n2):
    x1, y1 = n1
    x2, y2 = n2
    return ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)

def norma_inf(a, b):
    dist = -inf
    for i in range(len(a)):
        dist = max(abs(a[i] - b[i]), dist)
    return dist