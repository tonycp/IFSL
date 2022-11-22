from ._units import Unit
from enum import Enum

X_DIR = [0,1,1,1,0,-1,-1,-1]
Y_DIR = [1,1,0,-1,-1,-1,0,1]

DIRECTIONS = Enum('DIRECTIONS','N NE E SE S SW W NW')

STATES = Enum('STATES','Walking Swaping Attacking Stand')

def direction_to_int(direction):
    if direction == DIRECTIONS.N:
        return 0
    elif direction == DIRECTIONS.NE:
        return 1
    elif direction == DIRECTIONS.E:
        return 2
    elif direction == DIRECTIONS.SE:
        return 3
    elif direction == DIRECTIONS.S:
        return 4
    elif direction == DIRECTIONS.SW:
        return 5
    elif direction == DIRECTIONS.W:
        return 6
    elif direction == DIRECTIONS.NW:
        return 7

class StateMannager:
    class Connector:
        __id = 0
        def __init__(self,agent, unit:Unit, possition, functions:list[function]) -> None:
            self.agent = agent
            self.unit = unit 
            self.__current_hp = unit.GetHealthPoints
            self.__current_possition = possition
            StateMannager.Connector.__id += 1
            self.state = STATES.Stand
            self.timer = 0
            self.time_to_wait = 0
            self.__id = StateMannager.Connector.__id
            for func in functions:
                self.__dict__[func.__name__] = func

        def __update_possition(self,new_possition):
            self.__current_possition = new_possition

        def __update_health_points(self, new_hp):
            if self.is_connected():
                self.__current_hp = new_hp
                return True
            return False

        def get_position(self):
            return self.__current_possition

        def get_health_points(self):
            return self.__current_hp

        def is_connected(self):
            return self.__current_hp <= 0 
    
    def __init__(self, map, agents_units_possitions):
        self.map = map
        self.agents = []
        self.move_notifications = []
        self.swap_notifications = []
        self.attack_notifications = []
        for ag, units in agents_units_possitions:
            self.agents.append(ag)
            for unit,possition in units:
                connector = self.create_connector(unit,possition)
                ag.Connect(connector)
                
        
                
    def create_connector(self,agent,unit:Unit,possition):
        return StateMannager.Connector(agent,unit,possition,[self.move_notifier,self.swap_notifier,self.attack_notifier])

    def exec_round(self):
        # Pedir acciones a cada una de las IAs
        for agent in self.agents:
            # Calcular vision para este agente
            agent.decide()
        # Ejecutar las acciones
        log = self.exec_actions()
        return log
        pass

    def exec_actions(self):
        pass
    
    def move_notifier(self,connector,direction):
        num_direction = direction_to_int(direction)
        x, y = connector.get_position()
        if self.map[x + X_DIR[num_direction] , y + Y_DIR[num_direction]].crossable:
            self.move_notifications.append((connector,num_direction))

    def swap_notifier(self,connector,direction):
        num_direction = direction_to_int(direction)
        x, y = connector.get_position()
        if self.map[x + X_DIR[num_direction] , y + Y_DIR[num_direction]].crossable:
        
    def attack_notifier(self,connector,pos_x,pos_y):
        pass
    

    

     
    
    