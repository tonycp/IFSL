from math import inf
from .connector import StateMannager as S
from random import randint
from .utils import int_to_direction, validMove, DIRECTIONS, STATES, I_DIR, J_DIR
from IA.basicAlgorithms import astar_search_problem, RoadMap, best_first_search, breadth_first_search_problem
from IA._definitions import NodeTree, MoveVoronoiProblem, FindVoronoiVertex, path_states, expand

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
        self.roadmap = RoadMap(self.map, connector.unit)
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
            self.end_cost = dict([(node.state, path_states(node)[1:])
                                 for node in breadth_first_search_problem(end, findproblem_end)])

            vertexproblem = MoveVoronoiProblem(
                start_costs=self.start_costs, end_cost=self.end_cost, roadmap=self.roadmap, goals=[goal])
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
                connector, dir_tuple[(x_togo - x_position, y_togo - y_position)])
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

def norma2(n1, n2):
    x1, y1 = n1
    x2, y2 = n2
    return ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)