from ._units import Unit
class State:
    class Connector:
        __id = 0
        def __init__(self, unit:Unit, possition) -> None:
            self.unit = unit 
            self.current_hp= unit.GetHealthPoints()
            self.current_possition = possition
            __id +=1
            self.__id = __id

        def update_possition(self,new_possition):
            self.current_possition = new_possition

        def touch(self, action_name, **params):
            if self.__dict__[action_name](params):
                return 
    
    def move(self):
        pass

    
    def swap(self):
        pass
    
    
    def attack(self):
        pass
    
    