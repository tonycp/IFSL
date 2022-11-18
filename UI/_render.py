import sys
import pygame
from _grid import get_grid

class Render:
    def __init__(self, height: int=600, width: int=800) -> None:
        """
        inicializador de clase, crea un screen de dimensiones (width, height)

        height -> alto del screen, width -> ancho del screen
        """
        self.__screen = pygame.display.set_mode(size=(width, height))
        self.background = get_grid(height, width)

    def clean(self):
        """
        clean limpia toda la imagen que se muestra en el screen de pygame
        """
        self.__screen.fill(self.background)

    def start(self, condition) -> None:
        """
        start se encarga de empezar la simulación despues de que pygame haya iniciado

        condition -> condición de parada de la simulación
        """
        while condition:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            #################################
            #      GET STATE CODE HERE      #
            #################################
            self.clean()
            self.__screen.blit(ball, ballrect)
            pygame.display.flip()
