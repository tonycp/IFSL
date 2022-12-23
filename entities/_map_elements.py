class Celd:
    def  __init__(self, location) -> None:
        self.location = location
        self.__unit = None

    def set_unit(self, unit = None) -> bool:
        if not self.is_empty(): return False
        self.__unit = unit
        return True

    @property
    def get_unit(self):
        return self.__unit

    @property
    def is_empty(self) -> bool:
        return self.__dict__.get("unit") is None
    
    @property
    def crossable(self, unit): # que la unidad tenga una propiedad para saber si puede o no pasar segun la profundidad
        return True

class RiverCeld(Celd):

    def __init__(self, location, depth) -> None:
        self.depth = depth
        Celd.__init__(location)

class RoadCeld(Celd):
    pass

class GrassCeld(Celd):

    def __init__(self, location, height) -> None:
        self.height = height
        Celd.__init__(location)

class MountainCeld(Celd):

    def __init__(self, location, height) -> None:
        self.height = height
        Celd.__init__(location)

    @property
    def crossable(self, unit):
        return False

class WallCeld(Celd):

    def __init__(self, location, height) -> None:
        self.height = height
        Celd.__init__(location)

    @property
    def crossable(self, unit):
        return False