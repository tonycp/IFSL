class Unit:
    def __init__(self, vision, speed, attack_power, attack_range, unit_cost) -> None:
        self.vision = vision
        self.speed = speed
        self.attack_power = attack_power
        self.attack_range = attack_range
        self.unit_cost = unit_cost
    
    def run(self, action):
        pass

    def move(self, place):
        pass
    
    def attack(self, other):
        pass

class Fighter(Unit):
    
    @property
    def weakness(self, other):
        return True if other is Archer or other is Knight else False

class Archer(Unit):

    @property
    def weakness(self, other):
        return True if other is Unit else False

class Knight(Unit):

    @property
    def weakness(self, other):
        return True if other is Pikeman else False

class Pikeman(Unit):

    @property
    def weakness(self, other):
        return True if other is Archer or other is Fighter else False

class Explorer(Unit):

    @property
    def weakness(self, other):
        return True if other is Unit else False
