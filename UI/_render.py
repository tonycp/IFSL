import sys
import pygame
from _grid import get_grid
import numpy as np
from typing import Iterable


class Render:
    def __init__(self, condition, last_state: np.matrix = None, height: int = 600, width: int = 800) -> None:
        """
        inicializador de clase, crea un screen de dimensiones (width, height)

        height -> alto del screen, width -> ancho del screen
        """
        self.condition = condition
        self.__screen = pygame.display.set_mode(size=(width, height))
        self.last_state = get_grid(height, width) if last_state is None else last_state

    def __iter__(self) -> Iterable:
        """
        iterador que recorre la simulación después de que pygame haya iniciado

        return -> Iterable(States)
        """
        while self.condition:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            #################################
            #      GET STATE CODE HERE      #
            #################################
            yield self.last_state # new state

    def start(self) -> None:
        """
        start se encarga de recorrer la simulación hasta el final después de que pygame haya iniciado

        condition -> condición de parada de la simulación
        """
        for i in self:
            self.update(i)

    def update(self, state: np.matrix) -> bool:
        """
        update se encarga de actualizar la imagen que se muestra en el screen de pygame

        return -> bool
        """
        # for item1, item2 in zip(self.last_state, state):
        #     for value1, value2 in zip(item1, item2):
        #         if(value1 is not value2):
        #             #change frame
        return False

    def clean(self) -> None:
        """
        clean limpia toda la imagen que se muestra en el screen de pygame
        """
        self.__screen.fill((0, 0, 0))
