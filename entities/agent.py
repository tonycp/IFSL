from .connector import StateMannager as S
from random import randint
from .utils import int_to_direction, DIRECTIONS, STATES

    
class Agent:   
    __id = 0
    def __init__(self, ia, map) -> None:
        self.connectors = {}
        Agent.__id +=1
        self.id =Agent.__id 
        self.ia = ia
        self.map = map
    
    def connect(self, connector:S.Connector):
        self.connectors[connector._Connector__id]= connector
    
    def move_connectors(self):
        for id,c in self.connectors.items():
            if not c.is_connected(): 
                self.connectors.__delitem__(id)
            else:
                self.ia.get_move_for(c)
    
    def decide(self):
        self.move_connectors()
        

class RandomMove_IA:
    def get_move_for(self, connector:S.Connector):
        if(connector.state == STATES.Stand):
            dir = randint(0,7)
            connector.notify_move(connector, int_to_direction(dir))
        elif(connector.timer > 0):
            connector.notify_move(connector, connector.prev_dir)
        

                
            
             
        