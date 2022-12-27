from math import inf
from .connector import StateMannager as S
from ._units import Knight
from random import randint
from .utils import int_to_direction, DIRECTIONS, STATES, I_DIR, J_DIR, norma2
from IA.basicAlgorithms import astar_search_problem, RoadMap, breadth_first_search_problem, best_first_search
from IA._definitions import NodeTree, MoveVoronoiProblem, FindVoronoiVertex, path_states, expand
from IA.formations import *


I_DIR = [-1, -1, 0, 1, 1, 1, 0, -1]
J_DIR = [0, 1, 1, 1, 0, -1, -1, -1]

dir_tuple = {
    (-1, 0): DIRECTIONS.N,
    (-1, 1): DIRECTIONS.NE,
    (0, 1): DIRECTIONS.E,
    (1, 1): DIRECTIONS.SE,
    (1, 0): DIRECTIONS.S,
    (1, -1): DIRECTIONS.SW,
    (0, -1): DIRECTIONS.W,
    (-1, -1): DIRECTIONS.NW
}


class Agent(object):
    __id = 0

    def __init__(self, ia, map, goal = None) -> None:
        self.connectors = {}
        Agent.__id += 1
        self.id = Agent.__id
        self.ia = ia
        self.map = map
        self.goal = goal
        self.formations = []
        self.invalidated = []
        self.roadmap = RoadMap(self.map, Knight())

    def connect(self, connector: S.Connector):
        self.connectors[connector._Connector__id] = connector

    def asign_to_formation(self, formation, conectors):
        self.formations.append(self.ia(conectors,formation,moveIA= RoadMapMove_IA , roadmap = self.roadmap, goal = self.goal)) 
        
     
    def invalidations(self, invalidated_conectors):
        self.invalidated = invalidated_conectors  
    
    
    def move_connectors(self):
        for form in self.formations:
            form.get_move(self.invalidated)
            
        # for id, c in self.connectors.items():
        #     if not c.is_connected():
        #         self.connectors.__delitem__(id)
        #     else:
        #         if self.ia.goal is None and self.goal is not None:
        #             self.ia.set_goal(c, self.goal, self.roadmap)
        #         self.ia.get_move_for(c)

    def decide(self, view):
        self.move_connectors()

class RandomMove_IA(object):
    def __init__(self, map) -> None:
        self.map = map

    def get_move_for(self, connector: S.Connector):
        if(connector.state == STATES.Stand):
            dir = randint(0, 7)
            connector.notify_move(connector, int_to_direction(dir))
        elif(connector.timer > 0):
            connector.notify_move(connector, connector.prev_dir)


class RoadMapMove_IA(object):
    def __init__(self) -> None:
        self.goal = None

    def set_goal(self, connector: S.Connector, goal=None, roadmap= None):
        self.roadmap = roadmap
        self.goal = goal
        self.path = []
        if goal is not None:
            self.start = True
            position_start = connector.get_position()
            vertexs_state_start = self.roadmap.get_vertex_area(*position_start)
            findproblem_start = FindVoronoiVertex(self.roadmap, vertexs_state_start)

            start = NodeTree(state=position_start)
            self.start_costs = dict([(node.state, path_states(node)[1:]) 
                                 for node in breadth_first_search_problem(start, findproblem_start)])

            position_end = self.goal
            vertexs_state_end = self.roadmap.get_vertex_area(*position_end)
            findproblem_end = FindVoronoiVertex(self.roadmap, vertexs_state_end)

            end = NodeTree(state=goal)
            self.end_cost = dict([(node.state, path_states(node)[-2::-1])
                                 for node in breadth_first_search_problem(end, findproblem_end)])
            start_end = [state for state in self.end_cost]
            vertexproblem = MoveVoronoiProblem(
                start_costs=self.start_costs, end_cost=self.end_cost, start_end=start_end, roadmap=self.roadmap, goals=[goal])
            self.goalpath = path_states(
                astar_search_problem(start, vertexproblem,
                                     lambda n: norma2(n.state, end.state)))[1:]

    def get_move_for(self, connector: S.Connector):
        # if self.viewrange(self.goal, connector.get_position(), connector.unit.get_vision_radio):              # stay near goal
        #     self.steer_toward(connector)
        # elif self.viewrange(self.goalpath[0], connector.get_position(), connector.unit.get_vision_radio):    # set next subgoal as the target
        #     self.subgoal = self.goalpath.pop(0)
        # elif self.goal is not None:             # steer toward the target
        #     self.steer_toward(connector)
        if (len(self.goalpath) == 0 and len(self.path) == 0):
            self.goal = None
        if self.goal is None: return
        elif len(self.path) == 0 and self.viewrange(self.goalpath[0], connector.get_position(), connector.unit.get_vision_radio):    # set next subgoal as the target
            self.subgoal = self.goalpath.pop(0)
        self.steer_toward(connector)

    def steer_toward(self, connector: S.Connector):
        if len(self.path) > 0:
            return self.next_steep(connector)
        if len(self.roadmap.color[connector.get_position()]) >= 3:
            self.path = self.next_roadmap_path(connector.get_position())
        elif self.start:
            self.path = self.start_costs[self.subgoal]
            self.start = False
        elif self.goal == self.subgoal:
            self.path = self.end_cost[self.subgoal]
        self.next_steep(connector=connector)

    def next_steep(self, connector: S.Connector):
        if(connector.state == STATES.Stand):
            x_togo, y_togo = self.path.pop(0)
            x_position, y_position = connector.get_position()
            connector.notify_move(
                connector, (x_togo - x_position, y_togo - y_position))
        elif(connector.timer > 0):
            connector.notify_move(connector, connector.prev_dir)
    
    def next_roadmap_path(self, position):
        colors = set(self.roadmap.color[position]).intersection(self.roadmap.color[self.subgoal])

        findproblem_end = FindVoronoiVertex(self.roadmap, [self.subgoal])
        end = NodeTree(state=position)
        filter = lambda n: findproblem_end.is_goal(n.state)
        newexpand = lambda n: [] if len(colors.intersection(self.roadmap.color[n.state])) != 2 else expand(problem=findproblem_end, node=n)
        path = path_states(best_first_search(end, filter, newexpand))[1:]
        return path

    def viewrange(self, poss, actposs, vision):
        return norma2(poss, actposs) <= vision


class ForamtionMoveControl_IA(object):
    
    def __init__(self, conectors:dict, formation_shape: Formation, main_position = None, moveIA= None , roadmap = None, goal = None) -> None:
        if formation_shape.N-1 != len(conectors): return
        main_position = main_position or formation_shape.nodes[formation_shape.main].position
        self.fake_main = S.Connector(None, Knight(), main_position, [self.notify_move]) 
        self.formation_shape = formation_shape
        self.conectors = conectors
        self.direction = DIRECTIONS.N
        self.asignment ={}
        self.ia = moveIA()
        self.roadmap = roadmap
        self.time_to_wait = max([conector.unit.get_move_cost for conector in conectors.values()])
        self.time_waited = self.time_to_wait
        self.goal = goal
        self.positioned = False
        index = 0
        if main_position and formation_shape.nodes[formation_shape.main].position is None:
            formation_shape.set_in(*main_position)
        for i in conectors.values():
            if index == formation_shape.main: 
                index+=1
            self.asignment[i] = index
            index+=1 
    
    def get_move(self, invalidated):
        if self.positioned:
            if self.ia.goal is None and self.goal is not None:
                self.ia.set_goal(self.fake_main, self.goal, self.roadmap)

            if self.time_waited > self.time_to_wait and not invalidated: 
                self.time_waited = 0 
                self.ia.get_move_for(self.fake_main)

            else:
                self.time_waited +=1
                for key in self.asignment.keys():
                    if(key.state != STATES.Stand and key.timer > 0):
                        key.notify_move(key, key.prev_dir) 
                if invalidated: 
                    for conector, direct in invalidated:
                        conector.notify_move(conector, direct) 
                        self.time_to_wait = max(self.time_to_wait, conector.unit.get_move_cost)
        else: 
           from_to = [(connect.unit.get_position, self.formation_shape.nodes[form_n].position) for connect, form_n in self.asignment.items]
           #TODO Call method for formation here  
    
    def notify_move(self, conector, direction):
        self.formation_shape.move(*direction)
        i, j = self.fake_main.get_position()
        newposs = i + direction[0], j + direction[1]
        self.fake_main.update_position(newposs)
        for key in self.asignment.keys():
            if(key.state == STATES.Stand):
                key.notify_move(key, dir_tuple[direction])
            elif(key.timer > 0):
                key.notify_move(key, key.prev_dir) 