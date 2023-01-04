from IA.basicAlgorithms import RRAstar, RoadMap, whcastar_search
from entities.utils import DIRECTIONS
from IA.formations import Formation
from .low_layer import BasicAgent
from IA._definitions import *
from Agent.ia import *
import numpy as np


class MediumAgentMove:
    def __init__(self, agents: set[BasicAgent], formation: Formation, roadmap: RoadMap, events: dict, id) -> None:
        self.id = id
        self.dirt: DIRECTIONS = formation.dir
        self.events = events
        self.ia: Path_Move_IA = None
        self.formation = formation
        self.invalidate = set()
        self.available = set()
        self.roadmap = roadmap
        self.agents = agents
        self.subgoals = 0
        self.positioned: bool = None
        self.all_goals: list[bool] = None
        self.rrastar_list: list[RRAstar] = None
        self.from_to: list[tuple[int, int]] = None
        self.path: dict[BasicAgent, list[tuple[int, int]]] = None
        self.max_cost: int = max(map(lambda x: x.get_move_cost(), agents))
        self.time: int = self.max_cost
        self.low_events: dict = { 'is_invalid': self._is_invalid, 'end_task': self._end_task }

    def go_to_formation(self, poss = None, dirt = None):
        if dirt is not None:
            self.formation.rotate(dirt)
            self.dirt = dirt
        if poss is not None and self.formation.piss != poss:
            self.formation.set_in(*poss)

        start_poss = [u for u in self.agents]
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
            self.events['end_task'](self.id)
            return

        if not len(self.invalidate) and (new_goal or len(self.available) == len(self.agents)):
            self.available.clear()
            self.from_to, self.rrastar_list = MediumAgentMove.order(self.from_to, self.rrastar_list)
            self.path = whcastar_search(goals = self.from_to, roadmap=self.roadmap, rrastar_list=self.rrastar_list, w=8)
            self._notify_task(self.path.items())
        self._notify_move()

    def order(goals, rrastar_list):
        newgoals, newarrastar_list = [], []
        zip_goals = list(zip(goals, rrastar_list))
        zip_goals.sort(key=lambda x: x[0][0] == x[0][1])
        for goal, rrastar in zip_goals:
            newgoals.append(goal)
            newarrastar_list.append(rrastar)
        return newgoals, newarrastar_list

    def move_in_formation(self, poss = None):
        new_goal = False
        if type(self.ia) is not RoadMapMove_IA:
            new_goal = True
            self.ia = RoadMapMove_IA()
        if poss is not None and self.ia.goal != poss:
            self.ia.set_goal(self.formation.poss, poss, self.roadmap)

        self._in_formation(new_goal)
    
    def explore_in_formation(self, area, size):
        new_goal = False
        if type(self.ia) is not SliceMapMove_IA:
            new_goal = True
            self.ia = SliceMapMove_IA()
        if area is not None and size is not None and self.ia.goal != (area, size):
            self.ia.set_goal(self.formation.poss, (area, size), self.roadmap)

        self._in_formation(new_goal)
    
    def _in_formation(self, new_goal):
        if not len(self.invalidate) and (new_goal or len(self.available) == len(self.agents)):
            self.path = self.ia.get_move_for(self.formation.poss)
            if self.path:
                self.available.clear()
                self._notify_formation_task(self.path)
            return
        self._update_formation()
        self._notify_move()

    def get_info(self):
        return self.agents, self.formation

    def inform_view(self, view):
        oponents = set()
        for ag in self.agents:
            oponents = oponents.union(view(ag.get_position(), 10))
        return list(oponents)

    def _notify_formation_task(self, path: list[tuple]):
        x, y = self.formation.poss
        dirts = [(next_x - x, next_y - y) for next_x, next_y in path]
        for unit in self.agents:
            actions = []
            x, y = unit.get_position()
            prev_move = unit.get_position()
            for dx, dy in dirts:
                move = (x + dx, y + dy)
                if prev_move == move:
                    actions.append(("wait", move))
                else:
                    actions.append(("move", move))
                prev_move = move
            unit.set_action_list(actions, self.max_cost - unit.get_move_cost(), self.low_events)

    def _notify_task(self, paths: list[tuple[BasicAgent, tuple]]):
        for unit, path in paths:
            actions = []
            prev_move = unit.get_position()
            for move, _ in path:
                if prev_move == move:
                    actions.append(("wait", move))
                else:
                    actions.append(("move", move))
                prev_move = move
            unit.set_action_list(actions, self.max_cost - unit.get_move_cost(), self.low_events)

    def _update_formation(self):
        if self.time == 0 and not len(self.invalidate) and len(self.path):
            self.time = self.max_cost - 1
            self.formation.set_in(*self.path.pop(0))
        elif self.time == 0:
            self.time = self.max_cost - 1
        else:
            self.time -= 1

    def _notify_move(self):
        agents = self.agents
        for unit in agents:
            unit.eject_action()
        self.invalidate.clear()

    def _is_invalid(self, agent):
        self.invalidate.add(agent)

    def _end_task(self, agent):
        self.available.add(agent)
    
    def _is_dead(self, agent):
        self.agents.remove(agent)
        self.events['dead_unit'](agent, self.id)
        if not len(self.agents):
            self.events['dead_formation'](self.id)

class MediumAgentFigth:
    def __init__(self, agents, oponents, map, events: dict, id):
        self.id = id
        self.events = events
        self.agents = agents
        self.oponents = oponents
        self.strategie = FigthGame()
        self.available = agents
        self.map = map
        self.my_events = { 'is_invalid': self._is_invalid, 'end_task': self._end_task, 'is_dead': self._is_dead}
        
    def asign_figters(self, available_agents, oponents):
        result = []
        if not len(oponents):
            self.events['end_task'](self.id)
            return result
        best = oponents[0]
        for ag in available_agents:
            nearest = math.inf
            for oponent in oponents:
                actual = norma2(ag.get_position(), oponent)
                if actual < nearest:
                    nearest = actual
                    best = oponent
            result.append((ag,best))
        return result 
            
    def view(self, view):
        oponents = set()
        for ag in self.agents:
            oponents = oponents.union(view(ag.get_position(), 10)) 
        self.oponents = list(oponents)

    def _end_task(self, agent):
        self.available.add(agent)
        
    def _is_invalid(self, agent):
        self.available.add(agent)

    def _is_dead(self, agent):
        self.agents.remove(agent)
        self.events['dead_unit'](agent, self.id)
        if agent in self.available:
            self.available.remove(agent)
        if not len(self.agents):
            self.events['dead_formation'](self.id)
    
    def figth_in_formation(self):
        #Se asignan las parejas de lucha segun distancia entre los enemigos y los conectores que estan sin hacer nada
        recalc = self.asign_figters(self.available, self.oponents)
        bussy = {} 
        best_move = {}
        attacks = []
        wait = []
        while recalc: #mientras que haya parajas a las que recalcularle el minimax
            x, y = recalc.pop(0)
            state = State(self.map, x.connector, self.map[y].unit, bussy.get(x.connector.id) or []) #se crea un estado que contemple el estado actual de la pareja
            #en combate, el mapa y las ubicaciones en las que ya no tiene sentido que se paren
            
            value, action = h_alphabeta_search_solution(self.strategie, state, lambda x, y, z: z >= 3, self.strategie.heuristic)
            
            if action[0] == "wait":
                wait.append((x, action[1]))
            elif action[0] == "move": #si la accion es un move entonces hay que ver si no confluyen variosmove a una misma casilla
                bus =  bussy.get(x.connector.id) or []
                bus.append(action[1])
                bussy[x.connector.id] = bus

                move = best_move.get(action[1])
                if move is not None: #en caso que varios se muevan al mismo lugar se coge el de resultado mas prometedor
                    x0,y0,V = move
                    if V < value:
                        move = (x,y,value)
                        recalc.append((x0,y0))
                    elif V >= value:
                        recalc.append((x,y))
                else: move = (x,y,value)
                best_move[action[1]] = move
            elif action[0] == "attack":
                attacks.append((x,action[1])) #si es un ataque siempre se puede ejecutar 
        self.set_actions(best_move, attacks, wait)
        self.eject_actions()

    def set_actions(self, moves, attacks, wait):
        for x , pos in attacks:
            x.set_action_list([("attack", pos)], events = self.my_events, rithm= 1)
        for pos, (x,_,_) in moves.items():
            x.set_action_list([("move", pos)], events = self.my_events)
        for x, pos in wait:
            x.set_action_list([("wait", pos)], events = self.my_events, rithm= 1)
              
    def eject_actions(self):
        for agent in self.agents:
            agent.eject_action()   
    
    def inform_figth():
        pass  
    
    def inform_view():
        pass
    