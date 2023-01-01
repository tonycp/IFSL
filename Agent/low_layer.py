from entities.utils import STATES


class BasicAgent:
    
    def __init__(self, connector, action_list, rithm):
        self.connector = connector
        self.action_list = action_list 
        self.rithm = rithm
        self.current_time = rithm
        
    def set_action_list(self,action_list, rithm):
        self.action_list = action_list 
        self.rithm = rithm
        
            
    def inform_attack():
        """informar sobre ataques recibidos"""
        pass 
    
    def inform_move():
        """informar exito a fracaso en la accion de moverse"""
        pass
    
    def eject_action(self):
        """ejecutar una accion por el conector"""
        if self.current_time < self.rithm: 
            self.current_time+=1
            if(self.conector.state != STATES.stend and  self.connector.timer > 0):
                self.connector.notify_move(self.connector, self.connector.prev_dir)
            return 
        
        self.current_time = 0
        action = self.action_list[0]
        if action[0] == "attack":
            self.connector.notify_attack(self.connector, *action[1])
        elif action[0] == "move":
            self.connector.notify_attack(self.connector,  action[1] - self.connector.get_possition())
        
    
