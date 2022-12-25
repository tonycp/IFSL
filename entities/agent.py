from math import inf
from .connector import StateMannager as S
from random import randint
from .utils import int_to_direction, DIRECTIONS, STATES, I_DIR, J_DIR
from IA.basicAlgorithms import astar_search_problem, RoadMap, breadth_first_search_problem
from IA._definitions import NodeTree, MoveVoronoiProblem, FindVoronoiVertex, path_states
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
        self.ia = ia(map)
        self.map = map
        self.goal = goal

    def connect(self, connector: S.Connector):
        self.connectors[connector._Connector__id] = connector

    def move_connectors(self):
        for id, c in self.connectors.items():
            if not c.is_connected():
                self.connectors.__delitem__(id)
            else:
                if self.ia.goal is None and self.goal is not None:
                    self.ia.set_goal(c, self.goal)
                self.ia.get_move_for(c)

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
    def __init__(self, map) -> None:
        self.map = map
        self.goal = None

    def set_goal(self, connector: S.Connector, goal=None):
        self.roadmap = RoadMap(map, connector.unit)
        self.goal = goal
        if goal is not None:
            position = connector.get_position()
            self.start = True
            vertexs_state = self.roadmap.get_vertex_area(position)
            findproblem = FindVoronoiVertex(self.map, vertexs_state)

            start = NodeTree(state=position)
            self.start_costs = dict([(node.state, path_states(
                node)) for node in breadth_first_search_problem(start, findproblem)])

            end = NodeTree(state=goal)
            self.end_cost = dict([(node.state, path_states(node))
                                 for node in breadth_first_search_problem(end, findproblem)])

            vertexproblem = MoveVoronoiProblem(
                start_costs=self.start_costs, end_cost=self.end_cost, goals=[goal])
            self.goalpath = path_states(
                astar_search_problem(start, vertexproblem,
                                     lambda n: norma2(n.state, end.state)))

    def get_move_for(self, connector: S.Connector):
        if self.viewrange(self.goal, connector.get_position(), connector.unit.get_vision_radio):              # stay near goal
            self.steer_toward(connector)
        elif self.viewrange(self.goalpath[-1], connector.get_position(), connector.unit.get_vision_radio):    # set next subgoal as the target
            self.subgoal = self.goalpath.pop()
        elif self.goal is not None:             # steer toward the target
            self.steer_toward(connector)

    def steer_toward(self, connector: S.Connector):
        if len(self.path) > 0:
            return self.next_steep(connector)
        if self.start:
            self.path = self.start_costs[1]
            self.start = False
        elif self.goal == self.subgoal:
            self.path = self.end_cost[1]
        else:
            self.path = self.next_roadmap_path(connector.get_position())
        self.next_steep()

    def next_steep(self, connector: S.Connector):
        if(connector.state == STATES.Stand):
            x_togo, y_togo = self.path.pop(0)
            x_position, y_position = connector.get_position()
            connector.notify_move(
                self.connector, dir_tuple[(x_togo - x_position, y_togo - y_position)])
        elif(connector.timer > 0):
            connector.notify_move(self.connector, connector.prev_dir)
    
    def next_roadmap_path(self, position):
        path = []
        colors = set(self.roadmap.color[position]).intersection(self.roadmap.color[self.subgoal])
        while position != self.subgoal:
            x, y = position
            for x_dir, y_dir in dir_tuple.keys():
                if len(set(self.roadmap[x + x_dir, y + y_dir]).intersection(colors)) == 2 and (x + x_dir, y + y_dir) not in path:
                    newposition = x + x_dir, y + y_dir
                    break
            path.append(newposition)
            position = newposition
        return path

    def viewrange(self, poss, actposs, vision):
        return norma2(poss, actposs) <= vision

def norma2(n1, n2):
    x1, y1 = n1
    x2, y2 = n2
    return ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)


class ForamtionMoveControl_IA(object):
    def __init__(self, map, conectors, formation_shape: Formation) -> None:
        if formation_shape.N-1 != len(conectors): return
        self.map = map 
        self.formation_shape = formation_shape
        self.conectors = conectors
        self.direction = DIRECTIONS.N
        self.asignment ={}
        index = 0
        for i in conectors:
            if index == formation_shape.main: index+=1
            self.asignment[i] = index 
    
    
            
        
     
        
    
        
    