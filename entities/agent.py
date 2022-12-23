from .connector import StateMannager as S
from random import randint
from .utils import int_to_direction, DIRECTIONS, STATES

    
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

                
            
             
        