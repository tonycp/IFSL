import sys
import numpy as np
import pygame as pg
from entities import Cell
from typing import Iterable
from ._grid import get_grid, get_sprint, transform
from entities.connector import StateMannager
from entities.agent import *
from entities._units import *
from IA._definitions import *

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
        self.map = map if map is not None else get_grid(width, height)
        agent_1, units_1 = Agent(map, (7, 8)), [
            (Knight(), (0, 0)),
            (Knight(), (0, 1)),
            (Knight(), (0, map.shape[1] - 1)),
            (Knight(), (map.shape[0] - 1,map.shape[1] - 1)),
            (Knight(), (8,9)),
            (Knight(), (7,8))
        ]
        # agent_2, units_2 = Agent(map, (6, 6)), [
        #     (Knight(), (6, 6)),
        # ]

        self.last_state = StateMannager(self.map, [(agent_1, units_1)])
        formation_1 = TwoRowsFormation(6, (15, 8))
        # formation_2 = OneFormation((6, 6))
        
        agent_1.asign_to_formation(formation=formation_1, conectors=agent_1.connectors)
        # agent_2.asign_to_formation(formation=formation_2, conectors=agent_2.connectors)
        
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
