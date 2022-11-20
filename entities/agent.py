from .connector import StateMannager as S

class Agent:   
    __id = 0
    def __init__(self, ia) -> None:
        self.connectors = {}
        __id +=1
        self.__id =__id 
        self.ia = ia
    
    def connect(self, connector:S.Connector):
        self.connectors[connector.__id]= connector
    
    def move_connectors(self,map_copy):
        for id,c in self.connectors:
            if not c.is_connected(): 
                self.connectors.__delitem__(id)
            else:
                self.ia.get_move_for(c,map_copy)
        
    

                
            
             
        