class Unit:
    def __init__(self, attack_range, damage, vision_radio, health_points, move_cost) -> None:
        self.__attack_range = attack_range
        self.__damage = damage
        self.__vision_radio = vision_radio
        self.__health_points = health_points
        self.__move_cost = move_cost
    
    @property 
    def GetAttackRange(self):
        return self.__attack_range
    
    @property 
    def GetDamage(self):
        return self.__damage
    
    @property 
    def GetVisionRadio(self):
        return self.__vision_radio
    
    @property 
    def GetHealthPoints(self):
        return self.__health_points
    
    @property 
    def GetMoveCost(self):
        return self.__move_cost
    
 
    
class Fighter(Unit):
    
    def __init__(self):
        super().__init__(1,100,5,1000,6)
        
    @property
    def weakness(self, other):
        return True if other is Archer or other is Knight else False

class Archer(Unit):

    def __init__(self):
        super().__init__(15,50,20,1000,3)
        
    @property
    def weakness(self, other):
        return True if other is Unit else False

class Knight(Unit):

    def __init__(self):
        super().__init__(1,200,10,1000,1)
        
    @property
    def weakness(self, other):
        return True if other is Pikeman else False

class Pikeman(Unit):

    def __init__(self):
        super().__init__(2,50,5,1000,6)
        
    @property
    def weakness(self, other):
        return True if other is Archer or other is Fighter else False

class Explorer(Unit):

    def __init__(self):
        super().__init__(1,20,25,1000,2)
        
    @property
    def weakness(self, other):
        return True if other is Unit else False
