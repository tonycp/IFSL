import sys
import numpy as np
import pygame as pg
from entities import Cell
from typing import Iterable
from ._grid import get_grid, get_sprint, transform
from entities.connector import StateMannager
from entities.agent import *
from entities._units import Knight

class Render:
    def __init__(self, condition, map: np.matrix = None, width: int = 800, height: int = 600) -> None:
        """
        inicializador de clase, crea un screen de dimensiones (width, height)

        condition -> condición de parada,
        last_state -> estado inicial,
        height -> alto del screen, 
        width -> ancho del screen
        """
        self.condition = condition
        self.__screen = pg.display.set_mode(size=(width, height))
        agent = Agent(RandomMove_IA(), map)
        self.map = map if map is not None else get_grid(width, height)
        self.last_state = StateMannager(self.map, [(agent, [(Knight(), (0,0))])])
        
        self.__max_shape_size = max(self.map.shape[1], self.map.shape[0])
        self.__min_screen_size = min(self.__screen.get_size()[0], self.__screen.get_size()[1])

        self.__shape_correction = (self.__max_shape_size - self.map.shape[0]) / 2 , (self.__max_shape_size - self.map.shape[1]) / 2
        self.__screan_correction = (self.__screen.get_size()[0] - self.__min_screen_size) / 2 , (self.__screen.get_size()[1] - self.__min_screen_size) / 2
        
        self.__scale = int(self.__min_screen_size / self.__max_shape_size) + 1, int(self.__min_screen_size / self.__max_shape_size) + 1
        self.update(map.A1)

    def __iter__(self) -> Iterable[list[Cell]]:
        """
        iterador que recorre la simulación después de que pygame haya iniciado

        return -> Iterable[list[Cell]]
        """
        while self.condition:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()
            yield self.last_state.exec_round()  # new state

    def start(self, time: int = 3) -> None:
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
        for Cell in state:
            sprint = get_sprint(self.__scale, Cell)
            pos = transform(sprint, self.__scale, Cell.location, self.__shape_correction, self.__screan_correction)
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
