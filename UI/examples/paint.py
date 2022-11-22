import numpy as np
from entities import Celd, GrassCeld, MountainCeld, RoadCeld, RiverCeld, WallCeld


def paint_example_celd(i: int, j: int, value) -> Celd:
    if value == '_':
        return GrassCeld((i, j), 1)
    elif value == '*':
        return MountainCeld((i, j), 5)
    elif value == '.':
        return RoadCeld((i, j))
    elif value == '|':
        return RiverCeld((i, j), 3)
    elif value == '+':
        return WallCeld((i, j), 4)
    else:
        raise NotImplementedError("no existe ese elemento en el ejemplo")


def get_example_grid(example) -> np.matrix:
    return np.matrix([[paint_example_celd(i, j, example[j][i]) for i in range(len(line))] for line, j in zip(example, range(len(example)))], copy=False)
