import sys
import pygame as pg
from pygame.surface import Surface
import numpy as np
from entities import Celd
from typing import Iterable
from ._grid import get_grid, get_color

class Render:
    def __init__(self, condition, last_state: np.matrix = None, width: int = 800, height: int = 600) -> None:
        """
        inicializador de clase, crea un screen de dimensiones (width, height)

        height -> alto del screen, width -> ancho del screen
        """
        self.condition = condition
        self.__screen = pg.display.set_mode(size=(width, height))
        self.last_state = get_grid(
            width, height) if last_state is None else last_state
        self.__scale = self.__screen.get_size()[0] / self.last_state.shape[1], self.__screen.get_size()[1] / self.last_state.shape[0]
        self.update(last_state.A1)

    def __iter__(self) -> Iterable[list[Celd]]:
        """
        iterador que recorre la simulación después de que pygame haya iniciado

        return -> Iterable(States)
        """
        while self.condition:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()
            #################################
            #      GET STATE CODE HERE      #
            #################################
            yield self.last_state.A1  # new state

    def start(self) -> None:
        """
        start se encarga de recorrer la simulación hasta el final después de que pygame haya iniciado

        condition -> condición de parada de la simulación
        """
        for i in self:
            self.update(i)

    def blits(self, state: list[Celd]) -> None:
        return self.__screen.blits([(get_color(i), i.location) for i in state])

    def update(self, state: list[Celd]) -> bool:
        """
        update se encarga de actualizar la imagen que se muestra en el screen de pygame

        return -> bool
        """
        for celd in state:
            sprint = Surface(self.__scale)
            sprint.fill(get_color(celd))
            pos = celd.location[0] * self.__scale[0], celd.location[1] * self.__scale[1]
            sprint_rect = sprint.get_rect().move(pos)
            try:
                self.__screen.blit(sprint, sprint_rect)
            except:
                return False
        pg.display.update()
        return True

    def clean(self) -> None:
        """
        clean limpia toda la imagen que se muestra en el screen de pygame
        """
        self.__screen.fill((0, 0, 0))
