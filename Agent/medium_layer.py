from IA._definitions import *
class MediumAgentMove:
    def go_to_formation():
        pass
    
    def move_in_formation():
        pass
    
    def rotate_formation():
        pass
    
    def explore_in_frmation():
        pass 
     
    def inform_move():
        pass 
    
    def inform_view():
        pass
    
   
     
class MediumAgentFigth:
    def __init__(self, connectors, oponents):
        self.connectors = connectors
        self.oponents = oponents
        self.stategie = FigthGame()
        self.current_actions = []
        
    def asign_figters(self):
        pass 
    
    def figth_in_formation(self):
        recalc = self.asign_figters()
        bussy = {}
        best_move = {}
        attacks = []
        while recalc:
            x,y = recalc.pop(0)
            state = State(map, x,y, bussy[x.__id] or [])
            value ,action = h_alphabeta_search_solution(self.strategie,state, 3, self.strategie.heuristic)
            if action[0] == "move":
                bus =  bussy.get(x.__id) or []
                bus.append(action[1])
                bussy[x.__id] = bus 
                
                move = best_move.get(action[1]) or (x,y,value)
                x0,y0,V = move
                if V < value:
                    move = (x,y,value)
                    recalc.append((x0,y0))
                best_move[action[1]] = move
            elif action[0] == "attack":
                attacks.append((x,action[1]))
        self.set_actions(best_move, attacks)
                
                
    def set_actions(self, moves, attacks):
        pass   
        
    
    def inform_figth():
        pass 
    
    def inform_view():
        pass
    