import numpy as np
from entities import Cell, GrassCell, MountainCell, RoadCell, RiverCell, WallCell


def paint_example_Cell(i: int, j: int, value) -> Cell:
    if value == '_':
        return GrassCell((i, j), 1)
    elif value == '*':
        return MountainCell((i, j), 5)
    elif value == '.':
        return RoadCell((i, j))
    elif value == '|':
        return RiverCell((i, j), 3)
    elif value == '+':
        return WallCell((i, j), 4)
    else:
        raise NotImplementedError("no existe ese elemento en el ejemplo")


def get_example_grid(example) -> np.matrix:
    return np.matrix([[paint_example_Cell(i, j, example[i][j]) for j in range(len(line))] for line, i in zip(example, range(len(example)))], copy=False)
