import numpy as np
import pygame as pg
from pygame.surface import Surface
from entities import Celd, MountainCeld, RiverCeld, RoadCeld, WallCeld, GrassCeld

example = ['__________',
           '_***_.____',
           '_***_..___',
           '_****_.___',
           '__|__.....',
           '__|__.____',
           '__|||.||||',
           '_____.____',
           '_____.____',
           '_____.____']


def paint_empty_celd(i: int, j: int) -> Celd:
    return Celd((i, j))


def paint_example_celd(i: int, j: int) -> Celd:
    value = example[j][i]
    if value == '_':
        return GrassCeld((i, j), 1)
    elif value == '*':
        return MountainCeld((i, j), 5)
    elif value == '.':
        return RoadCeld((i, j))
    elif value == '|':
        return RiverCeld((i, j), 3)
    else:
        raise NotImplementedError("no existe ese elemento en el ejemplo")


def get_color(celd: Celd) -> pg.Color:
    """
    Dado un tipo de celda retorna su representante en color

    return -> pg.Color
    """
    if celd.is_empty:
        if (type(celd) is MountainCeld):
            return pg.Color(255, 196, 70)
        elif (type(celd) is RiverCeld):
            return pg.Color(116, 116, 255)
        elif (type(celd) is RoadCeld):
            return pg.Color(254, 221, 150)
        elif (type(celd) is WallCeld):
            return pg.Color(255, 251, 217)
        elif (type(celd) is GrassCeld):
            return pg.Color(109, 255, 123)
        else:
            raise NotImplementedError("celd type indefinido")
    else:
        return pg.Color(255, 0, 0)


def get_example_grid() -> np.matrix:
    return np.matrix([[paint_example_celd(i, j) for i in range(len(line))] for line, j in zip(example, range(len(example)))], copy=False)


def get_grid(width: int, height: int, paint=paint_empty_celd) -> np.matrix:
    """
    Crea una grilla a partir de las dimensiones que se pasan en el método

    height -> alto del grid, width -> ancho del grid, paint -> función que determina que celda va en x y

    return -> np.matrix
    """
    return np.matrix([[paint(i, j) for i in range(width)] for j in range(height)], copy=False)


# class Singleton:
#     instance = None

#     def __new__(cls, *args, **kwargs):
#         if not isinstance(cls.instance, cls):
#             cls.instance = object.__new__(cls, args, kwargs)
#         return cls.instance


# class SprintSurface(Surface):
#     def __init_subclass__(cls, color) -> None:
#         sprint = Surface.__init_subclass__(cls)
#         sprint.fill(color)
#         return sprint


# class MountainSurface(SprintSurface):
#     def __init_subclass__(cls, color) -> None:
#         return SprintSurface.__init_subclass__(cls, color)


# class RiverSurface(SprintSurface):
#     def __init_subclass__(cls, color) -> None:
#         return SprintSurface.__init_subclass__(cls, color)


# class RoadSurface(SprintSurface):
#     def __init_subclass__(cls, color) -> None:
#         return SprintSurface.__init_subclass__(cls, color)


# class WallSurface(SprintSurface):
#     def __init_subclass__(cls, color) -> None:
#         return SprintSurface.__init_subclass__(cls, color)


# class GrassSurface(SprintSurface):
#     def __init_subclass__(cls, color) -> None:
#         return SprintSurface.__init_subclass__(cls, color)
