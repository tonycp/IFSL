from IA._definitions import *
from IA.basicAlgorithms import RRAstar, RoadMap, whcastar_search
from IA.formations import Formation
from entities.utils import STATES, dir_tuple
from entities.connector import StateMannager
import numpy as np

class MediumAgentMove:
    def __init__(self) -> None:
        self.formation: Formation
        self.conectors: dict[int, StateMannager.Connector]
        self.from_to: list[tuple[int, int]]
        self.rrastar_list: list[RRAstar]
        self.all_goals: list[bool]
        self.positioned: bool
        self.roadmap: RoadMap
        self.path: dict[StateMannager.Connector, list[tuple[int, int]]]
        self.ia: Move_IA

    def go_to_formation(self, poss = None, dirt = None):
        if dirt is not None and self.dirt != dirt:
            self.formation.rotate(dirt)
        if poss is not None and self.formation.nodes[self.formation.main].position != poss:
            self.formation.set_in(*poss)

        start_poss = [c.get_position() for _, c in self.conectors.items()]
        new_from_to = self.formation.asign_formation(start_poss)
        if self.from_to != new_from_to:
            self.rrastar_list = np.ndarray(shape=(len(self.conectors)), dtype=RRAstar)
        self.from_to = new_from_to
        
        self.all_goals = list(map(lambda x: x[0].get_position() != x[1], self.from_to))
        self.positioned = not any(self.all_goals)
        if self.positioned:
            return

        if self.path is None or not len(self.path[self.from_to[0][0]]):
            self.path = whcastar_search(goals=self.from_to, roadmap=self.roadmap, rrastar_list=self.rrastar_list, w=8)
        for key, path in self.path.items():
            if(key.state == STATES.Stand):
                (next_x, next_y), _ = path.pop(0)
                move = next_x - key.get_position()[0], next_y - key.get_position()[1]
                if move == (0, 0):
                    continue
                dir = dir_tuple[move]
                key.notify_move(key, dir)
    
    def move_in_formation(self, poss = None, invalidated = None):
        if poss is not None and self.ia.goal != poss:
            self.ia.set_goal(self.formation.poss, poss, self.roadmap)

        if self.time_waited > self.time_to_wait and not invalidated:
            self.time_waited = 0 
            self.path = self.ia.get_move_for(self.formation.poss, 2) # cambiar vision
            if self.path:
                next_x, next_y = self.path.pop(0)
                x, y = self.formation.poss
                self._notify_move((next_x-x, next_y-y))
            return

        self.time_waited += 1
        for _, value in self.conectors.items():
            if(value.state != STATES.Stand and value.timer > 0):
                value.notify_move(value, value.prev_dir)
        if invalidated:
            for conector, direct in invalidated:
                conector.notify_move(conector, direct)
                self.time_to_wait = max(self.time_to_wait, conector.unit.get_move_cost)
    
    def _notify_move(self, direction):
        self.formation.move(*direction)
        for _, key in self.conectors.items():
            if(key.state == STATES.Stand):
                key.notify_move(key, dir_tuple[direction])
            elif(key.timer > 0):
                key.notify_move(key, key.prev_dir)
    
    def explore_in_formation():
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
    