from enum import Enum


X_DIR = [0,1,1,1,0,-1,-1,-1]
Y_DIR = [1,1,0,-1,-1,-1,0,1]

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

class Singleton:
    instance = None
    def __new__(cls,*args, **kwargs):
        if not isinstance(cls.instance,cls):
            cls.instance = object.__new__(cls)
        return cls.instance


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