from entities.utils import STATES
from entities.agent import *

class BasicAgent:
    
    def __init__(self, connector, action_list=None, events=None, rithm =None):
        self.connector = connector
        self.action_list = action_list
        self.rithm = rithm
        if rithm is None: 
            self.rithm = connector.unit.get_move_cost
        self.current_time = self.rithm
        self.events = events
        self.prev = None
        self.is_invalid =False
        
    def set_action_list(self,action_list = None, rithm = None, events = None):
        self.prev = None
        self.is_invalid =False
        self.action_list = action_list or self.action_list
        self.rithm = rithm or self.rithm
        self.events = events or self.events
        
    def invalid_move(self, move):
        self.is_invalid = True
        self.events['is_invalid'](self)


    def get_position(self):
        return self.connector.get_position()
    
    def get_move_cost(self):
        return self.connector.unit.get_move_cost 
          
    def eject_action(self):
        """ejecutar una accion por el conector"""
        if not self.connector.is_connected():
            self.events["id_dead"](self)
            return
            
        if self.is_invalid:
            self.is_invalid = False
            direction = dir_tuple[(self.prev[1][0] - self.connector.get_position()[0], self.prev[1][1] - self.connector.get_position()[1])]
            self.connector.notify_move(self.connector, direction)
            self.current_time = 0
            return
        
        if(self.prev is not None and self.prev[0]=="move" and self.connector.state != STATES.Stand and  self.connector.timer > 0):
            direction = dir_tuple[(self.prev[1][0] - self.connector.get_position()[0], self.prev[1][1] - self.connector.get_position()[1])]
            self.connector.notify_move(self.connector, direction)
        
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
                    direction = dir_tuple[(self.prev[1][0] - self.connector.get_position()[0], self.prev[1][1] - self.connector.get_position()[1])]
                    self.connector.notify_move(self.connector, direction)
            else: 
                self.events['end_task'](self)

    def __hash__(self):
        return hash(self.connector)
    
    def __eq__(self, __o: object):
        return __o is not None and self.connector.__id == __o.connector.__id
                