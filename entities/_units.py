from .utils import Singleton
from math import inf
class Unit:
    def __init__(self, attack_range, damage, vision_radio, health_points, move_cost) -> None:
        self.__attack_range = attack_range
        self.__damage = damage
        self.__vision_radio = vision_radio
        self.__health_points = health_points
        self.__move_cost = move_cost
    
    @property 
    def get_attack_range(self):
        return self.__attack_range
    
    @property 
    def get_damage(self):
        return self.__damage
    
    @property 
    def get_vision_radio(self):
        return self.__vision_radio
    
    @property 
    def get_health_points(self):
        return self.__health_points
    
    @property 
    def get_move_cost(self):
        return self.__move_cost
    
class Fighter(Unit, metaclass=Singleton):
    def __init__(self):
        Unit.__init__(self,1,100,5,1000,3)
        
    @property
    def weakness(self, other):
        return True if other is Archer or other is Knight else False

class Archer(Unit, metaclass=Singleton):

    def __init__(self):
        Unit.__init__(self,5,60,6,1000,2)
        
    @property
    def weakness(self, other):
        return True if other is Unit else False

class Knight(Unit, metaclass=Singleton):
    def __init__(self):
        Unit.__init__(self,1,120,5,1000,1)
        
    @property
    def weakness(self, other):
        return True if other is Pikeman else False

class Pikeman(Unit, metaclass=Singleton):

    def __init__(self):
        Unit.__init__(self,2,90,4,1000,3)
        
    @property
    def weakness(self, other):
        return True if other is Archer or other is Fighter else False

class Explorer(Unit, metaclass=Singleton):

    def __init__(self):
        Unit.__init__(self,1,40,7,1000,2)
        
    @property
    def weakness(self, other):
        return True if other is Unit else False

class Base(Unit, metaclass=Singleton):
    def __init__(self):
        Unit.__init__(self,5,100,7,8000, inf)
        
    @property
    def weakness(self, other):
        return False