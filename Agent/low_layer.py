from entities.utils import STATES
from entities.agent import *

class BasicAgent:
    
    def __init__(self, connector, action_list, rithm =None):
        self.connector = connector
        self.action_list = action_list
        self.rithm = rithm
        if rithm is None: 
            self.rithm = connector.unit.move_cost
        self.current_time = rithm
        
    def set_action_list(self,action_list = None, rithm = None):
        self.action_list = action_list or self.action_list
        self.rithm = rithm or self.rithm
         
    def get_position(self):
        return self.connector.get_position()
           
    def inform_attack():
        """informar sobre ataques recibidos"""
        pass 
    
    def inform_move():
        """informar exito o fracaso en la accion de moverse"""
        pass
    
    def eject_action(self):
        """ejecutar una accion por el conector"""
        
        if self.action_list:
            action = self.action_list.pop(0)
            if action[0] == "attack":
                self.connector.notify_attack(self.connector, *action[1])
                self.current_time = self.rithm
                
            elif action[0] == "move":
                if self.current_time < self.rithm: 
                    self.current_time+=1
                    if(self.conector.state != STATES.stend and  self.connector.timer > 0):
                        self.connector.notify_move(self.connector, self.connector.prev_dir)
                    return 

                self.current_time = 0
                direction = dir_tuple[(action[1][0] - self.connector.get_possition()[0], action[1][1] - self.connector.get_possition()[1])]
                self.connector.notify_attack(self.connector, direction)