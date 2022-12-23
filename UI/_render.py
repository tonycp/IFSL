import sys
import numpy as np
import pygame as pg
from entities import Cell
from typing import Iterable
from ._grid import get_grid, get_sprint, transform


class Render:
    def __init__(self, condition, last_state: np.matrix = None, width: int = 800, height: int = 600) -> None:
        """
        inicializador de clase, crea un screen de dimensiones (width, height)

        condition -> condición de parada,
        last_state -> estado inicial,
        height -> alto del screen, 
        width -> ancho del screen
        """
        self.condition = condition
        self.__screen = pg.display.set_mode(size=(width, height))
        self.last_state = get_grid(
            width, height) if last_state is None else last_state

        self.__shape = self.last_state.shape
        self.__max_shape_size = max(self.__shape[1], self.__shape[0])
        self.__min_screen_size = min(self.__screen.get_size()[0], self.__screen.get_size()[1])

        self.__shape_correction = (self.__max_shape_size - self.__shape[0]) / 2 , (self.__max_shape_size - self.__shape[1]) / 2
        self.__screan_correction = (self.__screen.get_size()[0] - self.__min_screen_size) / 2 , (self.__screen.get_size()[1] - self.__min_screen_size) / 2
        
        self.__scale = int(self.__min_screen_size / self.__max_shape_size) + 1, int(self.__min_screen_size / self.__max_shape_size) + 1
        self.update(last_state.A1)

    def __iter__(self) -> Iterable[list[Cell]]:
        """
        iterador que recorre la simulación después de que pygame haya iniciado

        return -> Iterable[list[Celd]]
        """
        while self.condition:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()
            #################################
            #      GET STATE CODE HERE      #
            #################################
            yield self.last_state.A1  # new state

    def start(self, time: int = 60) -> None:
        """
        start se encarga de recorrer la simulación hasta el final después de que pygame haya iniciado

        time -> cantidad de frames por segundo
        """
        clock = pg.time.Clock()
        for i in self:
            self.update(i)
            clock.tick(time)

    def update(self, state: list[Cell]) -> bool:
        """
        update se encarga de actualizar la imagen que se muestra en el screen de pygame

        return -> bool
        """
        for celd in state:
            sprint = get_sprint(self.__scale, celd)
            pos = transform(sprint, self.__scale, celd.location, self.__shape_correction, self.__screan_correction)
            try:
                self.__screen.blit(sprint, pos)
            except:
                return False
        pg.display.update()
        return True

    def clean(self) -> None:
        """
        clean limpia toda la imagen que se muestra en el screen de pygame
        """
        self.__screen.fill((0, 0, 0))
