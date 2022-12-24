from .connector import StateMannager as S
from random import randint
from .utils import int_to_direction, DIRECTIONS, STATES, I_DIR, J_DIR
from IA.basicAlgorithms import astar_tree_search, astar_search, RoadMap
from IA._definitions import NodeTree, Problem

class Agent:
    __id = 0

    def __init__(self, ia, map) -> None:
        self.connectors = {}
        Agent.__id += 1
        self.id = Agent.__id
        self.ia = ia
        self.map = map

    def connect(self, connector: S.Connector):
        self.connectors[connector._Connector__id] = connector

    def move_connectors(self):
        for id, c in self.connectors.items():
            if not c.is_connected():
                self.connectors.__delitem__(id)
            else:
                self.ia.get_move_for(c)

    def decide(self):
        self.move_connectors()


class RandomMove_IA:
    def get_move_for(self, connector: S.Connector):
        if(connector.state == STATES.Stand):
            dir = randint(0, 7)
            connector.notify_move(connector, int_to_direction(dir))
        elif(connector.timer > 0):
            connector.notify_move(connector, connector.prev_dir)


class RoadMapMove_IA(object):
    def __init__(self, map, connector: S.Connector) -> None:
        self.roadmap = RoadMap(map, connector.unit)

    def set_goal(self, connector: S.Connector, goal=None):
        self.goal = goal
        if goal is not None:
            problem = Problem(self.roadmap.distance[position], self.roadmap.vertex)
            position = connector.get_position()
            astar_search()

    def get_move_for(self, connector: S.Connector):
        if self.goal in self.viewrange:              # stay near goal
            self.stay_near_goal()
        elif self.goalpath[-1] in self.viewrange:    # set next subgoal as the target
            self.subgoal = self.goalpath.pop()
        elif self.goal is not None:             # steer toward the target
            self.steer_toward(connector)

    def steer_toward(self, connector: S.Connector):
        pass