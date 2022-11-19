from ._units import Unit

class StateMannager:
    class Connector:
        __id = 0
        def __init__(self, unit:Unit, possition, functions:list[function]) -> None:
            self.unit = unit 
            self.__current_hp= unit.GetHealthPoints()
            self.__current_possition = possition
            __id +=1
            self.__id = __id
            for func in functions:
                self.__dict__[func.__name__] = func

        def __update_possition(self,new_possition):
            self.__current_possition = new_possition

        def __update_health_points(self, new_hp):
            if self.is_connected():
                self.__current_hp = new_hp
                return True
            return False

        def get_health_points(self):
            return self.__current_hp

        def is_connected(self):
            return self.__current_hp <= 0 
    
    def __init__(self, map, agents_units_possitions):
        self.map = map
        self.connectors = []
        for ag, units in agents_units_possitions:
            for unit,possition in units:
                connector = self.create_connector(unit,possition)
                ag.Connect(connector)
                self.connectors.append(ag,connector)
        
                
    
    def create_connector(self,unit:Unit,possition):
        return self.Connector(unit,possition,[self.move,self.swap,self.attack])
    
    def move(self):
        pass

    def swap(self):
        pass
        
    def attack(self):
        pass
    

    

     
    
    