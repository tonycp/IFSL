from IA._definitions import *
from IA.basicAlgorithms import RRAstar, RoadMap, whcastar_search
from IA.formations import Formation
from entities.utils import DIRECTIONS, STATES, dir_tuple
from low_layer import BasicAgent
import numpy as np

class MediumAgentMove:
    def __init__(self) -> None:
        self.formation: Formation
        self.units: dict[int, BasicAgent]
        self.from_to: list[tuple[int, int]]
        self.rrastar_list: list[RRAstar]
        self.all_goals: list[bool]
        self.positioned: bool
        self.dirt: DIRECTIONS
        self.roadmap: RoadMap
        self.path: dict[BasicAgent, list[tuple[int, int]]]
        self.rithm: int
        self.ia: Move_IA
        self.invalidate: set
        self.subgoals = 0
        self.events: dict = { 'is_invalid': self._is_invalid, 'end_task': self._end_task, 'solve_invalid': self._solve_invalid }

    def go_to_formation(self, poss = None, dirt = None):
        if dirt is not None:
            self.formation.rotate(dirt)
            self.dirt = dirt
        if poss is not None and self.formation.piss != poss:
            self.formation.set_in(*poss)

        start_poss = [u.get_position() for _, u in self.units.items()]
        new_from_to = self.formation.asign_formation(start_poss)
        new_goal = False
        if self.from_to != new_from_to:
            self.rrastar_list = np.ndarray(shape=(len(self.units)), dtype=RRAstar)
            new_goal = True
        self.from_to = new_from_to
        
        self.all_goals = list(map(lambda x: x[0].get_position() != x[1], self.from_to))
        self.positioned = not any(self.all_goals)
        if self.positioned:
            [self.formation.dir]
            return

        if not len(self.invalidate) and (new_goal or self.subgoals):
            self.path = whcastar_search(goals=self.from_to, roadmap=self.roadmap, rrastar_list=self.rrastar_list, w=8)
            self._notify_task(self.path.items())
        self._notify_move()

    
    def move_in_formation(self, poss = None):
        if poss is not None and self.ia.goal != poss:
            self.ia.set_goal(self.formation.poss, poss, self.roadmap)

        if self.time_waited > self.time_to_wait:
            self.time_waited = 0 
            self.path = self.ia.get_move_for(self.formation.poss, 2) # cambiar vision
            if not self.path:
                return
            self._notify_formation_task()

        self.time_waited += 1
        for _, value in self.units.items():
            if(value.state != STATES.Stand and value.timer > 0):
                value.notify_move(value, value.prev_dir)
        if invalidated:
            for conector, direct in invalidated:
                conector.notify_move(conector, direct)
                self.time_to_wait = max(self.time_to_wait, conector.unit.get_move_cost)
    
    def _notify_formation_move(self, direction):
        self.formation.move(*direction)
        for unit in self.units.values():
            dir_tuple[direction]
            if(unit.state == STATES.Stand):
                unit.notify_move(unit, )
            elif(unit.timer > 0):
                unit.notify_move(unit, unit.prev_dir)
    
    def _is_invalid(self, id):
        self.invalidate.add(id)

    def _solve_invalid(self, id):
        self.invalidate.remove(id)

    def _end_task(self):
        self.subgoals += 1

    def _notify_formation_task(self):
        next_x, next_y = self.path.pop(0)
        x, y = self.formation.poss
        self.dirt = (next_x - x, next_y - y)
        for unit in self.units.values():
            x, y = unit.get_position()
            actions = [("move", (x + self.dirt[0], y + self.dirt[1]))]
            unit.set_action_list(actions, self.rithm, self.events)

    def _notify_task(self, paths):
        for unit, path in paths:
            actions = [("move", move) for move in path]
            unit.set_action_list(actions, self.rithm, self.events)

    def _notify_move(self):
        for unit in self.units.values():
            if unit not in self.invalidate: 
                continue
            unit.eject_action()

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
    