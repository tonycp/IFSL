class Cell:
    def  __init__(self, location) -> None:
        self.location = location
        self.__unit = None

    def set_unit(self, unit = None) -> bool:
        if unit is not None and not self.is_empty: return False
        self.__unit = unit
        return True

    @property
    def get_unit(self):
        return self.__unit

    @property
    def is_empty(self) -> bool:
        return self.__unit is None
    
    def crossable(self, unit): # que la unidad tenga una propiedad para saber si puede o no pasar segun la profundidad
        return True

class RiverCell(Cell):

    def __init__(self, location, depth) -> None:
        self.depth = depth
        Cell.__init__(self, location)

class RoadCell(Cell):
    pass

class GrassCell(Cell):

    def __init__(self, location, height) -> None:
        self.height = height
        Cell.__init__(self, location)

class MountainCell(Cell):

    def __init__(self, location, height) -> None:
        self.height = height
        Cell.__init__(self, location)

    def crossable(self, unit):
        return False

class WallCell(Cell):

    def __init__(self, location, height) -> None:
        self.height = height
        Cell.__init__(self, location)

    def crossable(self, unit):
        return False