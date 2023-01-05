from IA.formations import Formation
from .high_layer import HighAgent
from .low_layer import BasicAgent
from entities.connector import StateMannager as S

class Agent(object):
    __id = 0

    def __init__(self, map, roadmap, base) -> None:
        self.formations: dict[int, (list[BasicAgent], Formation)] = {}
        self.agents: dict[S.Connector, BasicAgent] = {}
        self.formation_count = 0
        Agent.__id += 1
        self.id = Agent.__id
        self.roadmap = roadmap
        self.map = map
        self.ia = None
        self.base = base

    def connect(self, connectors: list[S.Connector], formation):
        self.formation_count += 1
        formations = set()
        for connector in connectors:
            agent = BasicAgent(connector)
            formations.add(agent)
            self.agents[connector] = agent
        self.formations[self.formation_count] = (formations, formation)
     
    def invalidations(self, invalidated_conectors: list[tuple[S.Connector, tuple]]):
        for connector, move in invalidated_conectors:
            self.agents[connector].invalid_move(move)
    
    def disconnect(self, connector: S.Connector):
        self.agents.pop(connector).is_dead()

    def decide(self, view, filter1, filter2):
        if self.ia is None:
            self.ia = HighAgent(self.map, self.roadmap, self.formations, self.base)
        self.ia.decide(view, filter1, filter2)