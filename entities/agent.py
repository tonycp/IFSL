from .connector import StateMannager as S
from .connector import DIRECTIONS, STATES
from random import randint

def int_to_direction(direction):
    if direction == 0:
        return DIRECTIONS.N
    elif direction == 1:
        return DIRECTIONS.NE
    elif direction == 2:
        return DIRECTIONS.E
    elif direction == 3:
        return DIRECTIONS.SE
    elif direction == 4:
        return DIRECTIONS.S
    elif direction == 5:
        return DIRECTIONS.SW
    elif direction == 6:
        return DIRECTIONS.W
    elif direction == 7:
        return DIRECTIONS.NW
    
class Agent:   
    __id = 0
    def __init__(self, ia) -> None:
        self.connectors = {}
        Agent.__id +=1
        self.__id =Agent.__id 
        self.ia = ia
    
    def connect(self, connector:S.Connector):
        self.connectors[connector._Connector__id]= connector
    
    def move_connectors(self,map_copy):
        for id,c in self.connectors.items():
            if not c.is_connected(): 
                self.connectors.__delitem__(id)
            else:
                self.ia.get_move_for(c,map_copy)
        

class RandomMove_IA:
    def get_move_for(self,connector:S.Connector, map):
        if(connector.state == STATES.Stand):
            dir = randint(0,7)
            connector.move_notifier(connector,int_to_direction(dir))

                
            
             
        