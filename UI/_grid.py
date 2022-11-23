from typing import Union
import numpy as np
from pygame import Rect, Surface, Color
from entities import Celd, MountainCeld, RiverCeld, RoadCeld, WallCeld, GrassCeld


default = {
    'MountainCeld': Color(255, 196, 70),
    'RiverCeld': Color(116, 116, 255),
    'RoadCeld': Color(254, 221, 150),
    'WallCeld': Color(255, 251, 217),
    'GrassCeld': Color(109, 255, 123),
    'Celd': Color(255, 0, 0)
}

def paint_empty_celd(i: int, j: int) -> Celd:
    """
    dada una posición i, j crea una celda cualquiera

    i y j -> índices o posiciones de la celda en el grid

    return -> Celd
    """
    return Celd((i, j))


class Singleton(object):
    """
    clase usada para que todas las intancias hagan referencia al mismo objeto

    __instance -> unica instancia de la clase
    """
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '__instance'):
            cls.__instance = super().__new__(cls, *args, **kwargs)
        return cls.__instance


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


class MountainSurface(Singleton, SprintSurface):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


class RiverSurface(Singleton, SprintSurface):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


class RoadSurface(Singleton, SprintSurface):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


class WallSurface(Singleton, SprintSurface):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


class GrassSurface(Singleton, SprintSurface):
    def __init__(self, source: Union[Color, Surface], *args, **kwargs) -> None:
        return SprintSurface.__init__(self, source, *args, **kwargs)


def get_sprint(scale: tuple[int, int], celd: Celd, source: Union[Color, Surface] = None) -> Surface:
    """
    dado un tipo de celda retorna su representante en sprint

    scale -> escala del screen,
    celd -> celda a convertir en sprint,
    source -> representación de la celda

    return -> Surface
    """
    source = source if source is not None else default[type(celd).__name__]
    if celd.is_empty:
        if (type(celd) is MountainCeld):
            return MountainSurface(source, scale)
        elif (type(celd) is RiverCeld):
            return RiverSurface(source, scale)
        elif (type(celd) is RoadCeld):
            return RoadSurface(source, scale)
        elif (type(celd) is WallCeld):
            return WallSurface(source, scale)
        elif (type(celd) is GrassCeld):
            return GrassSurface(source, scale)
        else:
            raise NotImplementedError("celd type indefinido")
    else:
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


def get_grid(width: int, height: int, paint=paint_empty_celd) -> np.matrix:
    """
    Crea una grilla a partir de las dimensiones que se pasan en el método

    height -> alto del grid, 
    width -> ancho del grid, 
    paint -> función que determina que celda va en x y

    return -> np.matrix
    """
    return np.matrix([[paint(i, j) for j in range(width)] for i in range(height)], copy=False)
