import sys
import pygame

def start(condition) -> None:
    """
    Este método se encarga de empezar la simulación despues de que pygame haya iniciado

    condition -> condición de parada de la simulación
    """
    while condition:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        # more code
