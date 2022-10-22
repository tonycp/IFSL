class Celd:
    def  __init__(self, location, ) -> None:
        self.location = location

    @property
    def crossable(self, unit): # que la unidad tenga una propiedad para saber si puede o no pasar segun la profundidad
        return True

class RiverCeld(Celd):

    def __init__(self, location, depth) -> None:
        self.depth = depth
        super().__init__(location)

    @property
    def crossable(self, unit):
        return super().crossable

class RoadCeld(Celd):

    @property
    def crossable(self, unit):
        return super().crossable

class GrassCeld(Celd):

    def __init__(self, location, height) -> None:
        self.height = height
        super().__init__(location)

    @property
    def crossable(self, unit):
        return super().crossable

class MountainCeld(Celd):

    def __init__(self, location, height) -> None:
        self.height = height
        super().__init__(location)

    @property
    def crossable(self, unit):
        return False

class WallCeld(Celd):

    def __init__(self, location, height) -> None:
        self.height = height
        super().__init__(location)

    @property
    def crossable(self, unit):
        return False