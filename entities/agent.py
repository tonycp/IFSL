from math import inf
from .connector import StateMannager as S
from ._units import Knight
from random import randint
from .utils import int_to_direction, DIRECTIONS, STATES, I_DIR, J_DIR, norma2
from IA.basicAlgorithms import RRAstar, astar_search_problem, RoadMap, breadth_first_search_problem, best_first_search, whcastar_search
from IA._definitions import NodeTree, MoveVoronoiProblem, FindVoronoiVertex, path_states, expand, CSP_UncrossedAsignmentTime
from IA.formations import *
import numpy


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

    def __init__(self, map, goal = None) -> None:
        self.connectors = {}
        Agent.__id += 1
        self.id = Agent.__id
        self.map = map
        self.goal = goal
        self.formations = []
        self.invalidated = []
        self.roadmap = RoadMap(self.map, Knight())

    def connect(self, connector: S.Connector):
        self.connectors[connector._Connector__id] = connector

    def asign_to_formation(self, formation, conectors):
        self.formations.append(ForamtionMoveControl_IA(conectors, formation, moveIA= RoadMapMove_IA, roadmap = self.roadmap, goal = self.goal)) 
        
     
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
        return self.path

    def steer_toward(self, connector: S.Connector):
        if len(self.roadmap.color[connector.get_position()]) >= 3:
            self.path = self.next_roadmap_path(connector.get_position())
        elif self.start:
            self.path = self.start_costs[self.subgoal]
            self.start = False
        elif self.goal == self.subgoal:
            self.path = self.end_cost[self.subgoal]
    
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
        self.from_to = None
        self.ia = moveIA()
        self.roadmap = roadmap
        self.time_to_wait = max([conector.unit.get_move_cost for conector in conectors.values()])
        self.time_waited = self.time_to_wait
        self.goal = goal
        self.positioned = False
        self.result = None
        # index = 0
        self.rrastar_list = numpy.ndarray(shape=(len(conectors)), dtype=RRAstar)
           
        if main_position and formation_shape.nodes[formation_shape.main].position is None:
            formation_shape.set_in(*main_position)
        # for i in conectors.values():
        #     if index == formation_shape.main: 
        #         index+=1
        #     self.asignment[i] = index
        #     index+=1 
    
    def get_move(self, invalidated):
        if self.positioned:
            self.move_in_formation(invalidated)
        else:
            
            self.go_to_formation()

    def asign_formation(self):
        csp = CSP_UncrossedAsignmentTime([c for _,c in self.conectors.items()], [pos.position for pos in self.formation_shape.nodes if pos.state != self.formation_shape.main])
        csp.back_track_solving()
        if csp.asignment: 
            from_to = [(y,x) for x,y in csp.asignment.items()]     
        else:
            from_to = [(c, self.formation_shape.nodes[id].position) for id ,c in self.conectors.items()]
        return from_to

    def move_in_formation(self, invalidated):
        if self.ia.goal is None and self.goal is not None:
            self.ia.set_goal(self.fake_main, self.goal, self.roadmap)

        if  self.time_waited > self.time_to_wait and not invalidated: 
            self.time_waited = 0 
            self.result = self.ia.get_move_for(self.fake_main)
            if not self.result:
                if self.result == None:
                    self.rotate_formation(4)
                return
            next_x, next_y = self.result.pop(0)
            x, y = self.formation_shape.nodes[self.formation_shape.main].position
            self.notify_move((next_x-x, next_y-y))
        
        else:
            self.time_waited +=1
            for _,key in self.conectors.items():
                if(key.state != STATES.Stand and key.timer > 0):
                    key.notify_move(key, key.prev_dir) 
            if invalidated: 
                for conector, direct in invalidated:
                    conector.notify_move(conector, direct) 
                    self.time_to_wait = max(self.time_to_wait, conector.unit.get_move_cost)



    
    def go_to_formation(self):
        if not self.from_to:
            self.from_to = self.asign_formation()
        all_goals = list(map(lambda x: x[0].get_position() != x[1], self.from_to))
        self.positioned = not any(all_goals)
        if not self.positioned and (self.result is None or not len(self.result[self.from_to[0][0]])):
            self.result = whcastar_search(goals=self.from_to, roadmap=self.roadmap, rrastar_list=self.rrastar_list, w=8)
            
        elif not self.positioned:
            for key, path in self.result.items():
                if(key.state == STATES.Stand):
                    (next_x, next_y), _ = path.pop(0)
                    move = next_x - key.get_position()[0], next_y - key.get_position()[1]
                    if move == (0, 0):
                        continue
                    dir = dir_tuple[move]
                    key.notify_move(key, dir)             
    
    def rotate_formation(self, direction):
        self.positioned = False
        self.rrastar_list = numpy.ndarray(shape=(len(self.conectors)), dtype=RRAstar)
        self.from_to = None
        self.formation_shape.rotate(direction)
        self.go_to_formation()


    def notify_move(self, direction):
        self.formation_shape.move(*direction)
        i, j = self.fake_main.get_position()
        newposs = i + direction[0], j + direction[1]
        self.fake_main.update_position(newposs)
        for _, key in self.conectors.items():
            if(key.state == STATES.Stand):
                key.notify_move(key, dir_tuple[direction])
            elif(key.timer > 0):
                key.notify_move(key, key.prev_dir) 