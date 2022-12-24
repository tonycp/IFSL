from typing import Union
import numpy as np
from pygame import Rect, Surface, Color
from entities import Cell, MountainCell, RiverCell, RoadCell, WallCell, GrassCell, Singleton


default = {
    'MountainCell': Color(255, 196, 70),
    'RiverCell': Color(116, 116, 255),
    'RoadCell': Color(254, 221, 150),
    'WallCell': Color(255, 251, 217),
    'GrassCell': Color(109, 255, 123),
    'Team-1': Color(255, 0, 0),
    'Team-2': Color(0, 0, 255)
}

def paint_empty_Cell(i: int, j: int) -> Cell:
    """
    dada una posición i, j crea una Celda cualquiera

    i y j -> índices o posiciones de la Celda en el grid

    return -> Cell
    """
    return Cell((i, j))

class SprintSurface(Surface):
    """
    clase que representa un sprint de una superficie a pintar
    """
    def __new__(cls, source: Union[Color, Surface], *args, **kwargs) -> None:
        return Surface.__new__(cls, *args, **kwargs)

    def __init_subclass__(cls) -> None:
        return Surface.__init_subclass__()

    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        Surface.__init__(self, *args, **kwargs)
        if type(source) is Color:
            self.fill(source)
        elif type(source) is Surface:
            self.blit(source, source.get_rect())
        else:
            raise TypeError("source debe ser Color o Surface")


class MountainSurface(SprintSurface, metaclass=Singleton):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


class RiverSurface(SprintSurface, metaclass=Singleton):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


class RoadSurface(SprintSurface, metaclass=Singleton):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


class WallSurface(SprintSurface, metaclass=Singleton):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


class GrassSurface(SprintSurface, metaclass=Singleton):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


def get_sprint(scale: tuple[int, int], Cell: Cell, source: Union[Color, Surface] = None) -> Surface:
    """
    dado un tipo de Celda retorna su representante en sprint

    scale -> escala del screen,
    Cell -> Celda a convertir en sprint,
    source -> representación de la Celda

    return -> Surface
    """
    if Cell.is_empty:
        source = source if source is not None else default[type(Cell).__name__]
        if (type(Cell) is MountainCell):
            return MountainSurface(source, scale)
        elif (type(Cell) is RiverCell):
            return RiverSurface(source, scale)
        elif (type(Cell) is RoadCell):
            return RoadSurface(source, scale)
        elif (type(Cell) is WallCell):
            return WallSurface(source, scale)
        elif (type(Cell) is GrassCell):
            return GrassSurface(source, scale)
        else:
            raise NotImplementedError("Cell type indefinido")
    else:
        source = source if source is not None else default[f'Team-{Cell.get_unit.agent.id}']
        return SprintSurface(source, scale)


def transform( sprint: Surface, 
               scale: tuple[int, int], 
               location: tuple[int, int], 
               shape_correction: tuple[float, float] = (0, 0),
               scale_correction: tuple[float, float] = (0, 0)
               ) -> Rect:
    """
    traslada un Surface a una posición escalada para su representación en el screen de pygame

    sprint -> superficie a mover,
    scale -> escala del screen,
    location -> posición relativa del sprint,
    shape_correction -> corrección relativa a la posición,
    scale_correction -> corrección relativa al screen

    return -> Rect
    """
    return sprint.get_rect().move((location[1] + shape_correction[1]) * scale[0] + scale_correction[0], 
                                  (location[0] + shape_correction[0]) * scale[1] + scale_correction[1])


def get_grid(width: int, height: int, paint=paint_empty_Cell) -> np.matrix:
    """
    Crea una grilla a partir de las dimensiones que se pasan en el método

    height -> alto del grid, 
    width -> ancho del grid, 
    paint -> función que determina que Celda va en x y

    return -> np.matrix
    """
    return np.matrix([[paint(i, j) for j in range(width)] for i in range(height)], copy=False)
