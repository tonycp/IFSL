from entities.utils import STATES
from entities.agent import *

class BasicAgent:
    
    def __init__(self, connector, action_list, events, rithm =None):
        self.connector = connector
        self.action_list = action_list
        self.rithm = rithm
        if rithm is None: 
            self.rithm = connector.unit.move_cost
        self.current_time = rithm
        self.events = events
        self.prev = None
        self.is_invalid =False
        
    def set_action_list(self,action_list = None, rithm = None, events = None):
        self.is_invalid =False
        self.action_list = action_list or self.action_list
        self.rithm = rithm or self.rithm
        self.events = events or self.events
        
    def invalid_move(self, move):
        self.is_invalid = True
        self.events._is_invalid(self,move)
    
         
    def get_position(self):
        return self.connector.get_position()
           
    def eject_action(self):
        """ejecutar una accion por el conector"""
        if self.is_invalid:
            self.is_invalid = False
            self.connector.notify_move(self.connector, self.connector.prev_dir)
            self.current_time = 0
            return
        
        if(self.prev[0]=="move" and self.conector.state != STATES.stend and  self.connector.timer > 0):
            self.connector.notify_move(self.connector, self.connector.prev_dir)
        
        if self.current_time < self.rithm: 
            self.current_time+=1
        
        if  self.current_time == self.rithm:        
            if self.action_list: 

                self.prev = self.action_list.pop(0)

                if self.prev[0] == "attack":
                    self.connector.notify_attack(self.connector, *self.prev[1])
                    self.current_time = self.rithm

                elif self.prev[0] == "move":
                    self.current_time = 0
                    direction = dir_tuple[(self.prev[1][0] - self.connector.get_possition()[0], self.prev[1][1] - self.connector.get_possition()[1])]
                    self.connector.notify_attack(self.connector, direction)
            else: 
                self.events._end_task(self)
                