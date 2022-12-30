from cmath import inf
from queue import Queue
from Agent.low_layer import BasicAgent
from Agent.medium_layer import MediumAgentMove, MediumAgentFigth
from IA.basicAlgorithms import RoadMap
from entities.connector import StateMannager as S
from IA.formations import Formation, OneFormation, TwoRowsFormation
from random import shuffle

class HighAgent:
    percent_explorer = 0.25
    def __init__(self, map, roadmap: RoadMap, formations: dict[int, tuple[set, Formation]], base):
        self.map = map
        self.roadmap = roadmap
        self.formations = formations
        self.explorer = []
        self.troops = {}
        self.attack_troop ={}
        self.asing = None
        self.start = True
        self.in_misison:dict[int, Queue]={}
        self.base = base
        self.exp_formation = None
        self.exp_agent_formation = None
        self.exp_mision_formation = None

        for id, (agents, formation) in self.formations.items():
            self.troops[id] = MediumAgentMove(agents, formation, self.roadmap)
        # self.def_explorer()

    def decide(self, view):
        for id, troop in list(self.troops.items()):
            troop_view = troop.inform_view(view)
            if len(troop_view) > 0:
                self.set_figth(id, troop, troop_view)
                
        if self.exp_mision_formation is not None:
            if len(self.exp_mision_formation) == 1:
                self.exp_agent_formation.move_in_formation(self.exp_mision_formation.get())
            else:
                self.exp_agent_formation.go_to_formation(self.exp_mision_formation.get())
        # else:
        #     for id, (area, size) in self.asing.items():
        #         self.troops[id].explore_in_formation(area, size, self.roadmap)
        for id, move in self.in_misison.items():
            troop = self.troops.get(id)
            if troop is not None:
                troop.move_in_formation(move)
            else:
                self.attack_troop[id].view(view)
                self.attack_troop[id].figth_in_formation()

    def go_to_enemies(self, positions):
        if self.exp_formation is None and len(self.explorer):
            other_explorer = set()
            for id, _ in self.explorer:
                mision = self.in_misison.pop(id)
                self.troops.pop(id)
                self.asing.pop(id)
                if positions not in mision:
                    other_explorer.union(self.formations[id][0])
            
            self.exp_formation = TwoRowsFormation(len(other_explorer), self.base)
            self.exp_agent_formation = MediumAgentMove(other_explorer, self.exp_formation, self.roadmap)
            self.exp_mision_formation = Queue([self.base, positions])
        for id in self.troops.keys():
            mision = self.in_misison.get(id)
            if mision is not None and positions not in mision:
                self.in_misison[id].put(positions)

    def def_explorer(self):
        explorer = list(filter(lambda x: len(x[1][0]) == 1, self.formations.items()))
        explorer.sort(key=lambda x: x[1][0].get_move_cost())
        explorer_len = min(len(explorer), self.roadmap.color_num) # arreglar
        self.explorer = explorer[:explorer_len]
        self.asing = {}
        asing_num = list(map(lambda x: x[0], self.explorer))

        for num in asing_num:
            self.asing[num] = [set(), None]
        count = 0

        for area, size in self.roadmap.size.items():
            index = count % explorer_len
            if index == 0:
                shuffle(asing_num)
            asing = self.asing[index]
            asing[0].add(area)
            asing[1] = HighAgent.update_size(asing[2], size)
            count += 1

    def update_size(size_1, size_2):
        if size_1 is None:
            return size_2
        elif size_2 is None:
            return size_1
        max_size_x, min_size_x = max(size_1[0][0], size_2[0][0]), min(size_1[0][1], size_2[0][1])
        max_size_y, min_size_y = max(size_1[1][0], size_2[1][0]), min(size_1[1][1], size_2[1][1])
        return (max_size_x, min_size_x), (max_size_y, min_size_y)
        
    def get_view():
        pass

    def get_movement_information():
        pass

    def get_attack_information():

        pass

    def set_move(self, formation):
        pass

    def set_figth(self, id, troop, enemy):
        self.in_misison[id] = Queue()
        agent, _ = self.troops.pop(id).get_info()
        self.attack_troop[id] = MediumAgentFigth(agent, enemy, self.map)
        self.attack_troop[id].figth_in_formation()
        self.go_to_enemies(troop.formation.poss)