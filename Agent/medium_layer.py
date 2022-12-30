from IA._definitions import *
from IA.basicAlgorithms import RRAstar, RoadMap, whcastar_search
from entities.agent import RoadMapMove_IA
from IA.formations import Formation
from entities.utils import DIRECTIONS, STATES, dir_tuple
from low_layer import BasicAgent
import numpy as np


class MediumAgentMove:
    def __init__(self, agents: set[BasicAgent], formation: Formation, roadmap: RoadMap) -> None:
        self.dirt: DIRECTIONS = formation.dir
        self.ia = RoadMapMove_IA() # Move_IA
        self.formation = formation
        self.invalidate = set()
        self.roadmap = roadmap
        self.agents = agents
        self.subgoals = 0
        self.positioned: bool = None
        self.all_goals: list[bool] = None
        self.rrastar_list: list[RRAstar] = None
        self.from_to: list[tuple[int, int]] = None
        self.path: dict[BasicAgent, list[tuple[int, int]]] = None
        self.max_cost: int = max(agents, key=lambda x: x.get_move_cost())
        self.events: dict = { 'is_invalid': self._is_invalid, 'end_task': self._end_task, 'solve_invalid': self._solve_invalid }

    def go_to_formation(self, poss = None, dirt = None):
        if dirt is not None:
            self.formation.rotate(dirt)
            self.dirt = dirt
        if poss is not None and self.formation.piss != poss:
            self.formation.set_in(*poss)

        start_poss = [u.get_position() for u in self.agents]
        new_from_to = self.formation.asign_formation(start_poss)
        self.go_to_positions(new_from_to)


    def go_to_positions(self, from_to):
        new_goal = False
        if self.from_to != from_to:
            self.rrastar_list = np.ndarray(shape=(len(self.agents)), dtype=RRAstar)
            new_goal = True
        self.from_to = from_to
        self.all_goals = list(map(lambda x: x[0].get_position() != x[1], self.from_to))
        self.positioned = not any(self.all_goals)
        if self.positioned:
            return

        if not len(self.invalidate) and (new_goal or self.subgoals):
            self.path = whcastar_search(goals = self.from_to, roadmap=self.roadmap, rrastar_list=self.rrastar_list, w=8)
            self._notify_task(self.path.items())
        self._notify_move()


    def move_in_formation(self, poss = None):
        if poss is not None and self.ia.goal != poss:
            self.ia.set_goal(self.formation.poss, poss, self.roadmap)

        if not len(self.invalidate) and self.subgoals:
            path = self.ia.get_move_for(self.formation.poss, 2) # cambiar vision
            if path:
                self._notify_formation_task(path)
            return
        self._notify_move()

    def explore_in_formation():
        pass 
     
    def inform_move():
        pass 
    
    def inform_view():
        pass

    def _notify_formation_task(self, path: list[tuple]):
        x, y = self.formation.poss
        dirts = [(next_x - x, next_y - y) for next_x, next_y in path]
        for unit in self.agents.values():
            x, y = unit.get_position()
            actions = [("move", (x + dx, y + dy)) for dx, dy in dirts]
            unit.set_action_list(actions, self.rithm, self.events)

    def _notify_task(self, paths: list[tuple[BasicAgent, tuple]]):
        for unit, path in paths:
            actions = [("move", move) for move in path]
            unit.set_action_list(actions, self.rithm, self.events)

    def _notify_move(self):
        agents = self.agents
        if len(self.invalidate):
            agents = self.invalidate
        for unit in agents:
            unit.eject_action()

    def _is_invalid(self, agent):
        self.invalidate.add(agent)

    def _solve_invalid(self, agent):
        self.invalidate.remove(agent)

    def _end_task(self):
        self.subgoals += 1
    
    def _is_dead(self, agent):
        self.agents.remove(agent)

class MediumAgentFigth:
    def __init__(self, agents, oponents, map):
        self.agents = agents
        self.oponents = oponents
        self.stategie = FigthGame()
        self.available = agents
        self.my_events = { 'is_invalid': self._is_invalid, 'end_task': self._end_task, 'is_dead': self._is_dead}
        
    def asign_figters(self, available_agents, oponents):
        result = []
        best = oponents[0]
        for ag in available_agents:
            nearest = math.inf
            for oponent in oponents:
               if norma2(ag.get_position(), oponent.get_position()) < nearest:
                   best = oponent
            result.append((ag,best))
        return result 
            
    def view(self, oponents):
        self.oponents = oponents 
        
    def _end_task(self, agent):
        self.available.append(agent)
        
        
    def _is_invalid(self, agent):
        self.available.append(agent)

    def _is_dead(self, agent):
        self.agents = self.agents.intersection(agent)
        pass
    
    def figth_in_formation(self):
        recalc = self.asign_figters(self.available, self.oponents)
        bussy = {}
        best_move = {}
        attacks = []
        while recalc:
            x,y = recalc.pop(0)
            state = State(self.map, x.connector,y, bussy[x.__id] or [])
            value ,action = h_alphabeta_search_solution(self.strategie,state, 3, self.strategie.heuristic)
            if action[0] == "move":
                bus =  bussy.get(x.connector.__id) or []
                bus.append(action[1])
                bussy[x.connector.__id] = bus 
                
                move = best_move.get(action[1]) or (x,y,value)
                x0,y0,V = move
                if V < value:
                    move = (x,y,value)
                    recalc.append((x0,y0))
                best_move[action[1]] = move
            elif action[0] == "attack":
                attacks.append((x,action[1]))
        self.set_actions(best_move, attacks)
        self.eject_actions()
    
                
                
    def set_actions(self, moves, attacks):
        for x , pos in attacks:
            x.set_action_list([("attack", pos)], events = self.my_events)
        for pos, (x,_,_) in moves.items():
            x.set_action_list([("move", pos)], events = self.my_events)
              
    def eject_actions(self):
        for agent in self.agents:
            agent.eject_action()   
    
    def inform_figth():
        pass  
    
    def inform_view():
        pass
    