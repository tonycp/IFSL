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
        self.low_events = { "dead_formation": self._dead_formation, "dead_unit": self._dead_unit, "end_task": self._end_task }

        for id, (agents, formation) in self.formations.items():
            self.troops[id] = MediumAgentMove(agents, formation, self.roadmap, self.low_events, id)
            self.in_misison[id] = Queue()
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
            poss, action = move.queue[0] if len(move.queue) else (None, None)
            if action == "fight":
                self.attack_troop[id].view(view)
                self.attack_troop[id].figth_in_formation()
            elif action == "move_in_formation":
                self.troops[id].move_in_formation(poss)
            elif action == "go_to_formation":
                self.troops[id].go_to_formation()

    def go_to_enemies(self, positions):
        if self.exp_formation is None and len(self.explorer):
            other_explorer = set()
            for id, _ in self.explorer:
                mision = self.in_misison.pop(id)
                self.troops.pop(id)
                self.asing.pop(id)
                if positions not in map(lambda x: x[0], mision.queue):
                    other_explorer.union(self.formations[id][0])
            
            self.exp_formation = TwoRowsFormation(len(other_explorer), self.base)
            self.exp_agent_formation = MediumAgentMove(other_explorer, self.exp_formation, self.roadmap, self.low_events, -1)
            self.exp_mision_formation = Queue()
            self.exp_mision_formation.put((self.base, "go_to_formation"))
            self.exp_mision_formation.put((positions, "move_in_formation"))

        for id in self.troops.keys():
            mision = self.in_misison.get(id)
            if mision is not None and positions not in map(lambda x: x[0], mision.queue):
                self.in_misison[id].put((positions, "move_in_formation"))

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

    def _dead_formation(self, id):
        if id in self.in_misison:
            self.in_misison.pop(id)
        troop = self.troops if self.troops.get(id) else self.attack_troop
        if troop.get(id):
            troop.pop(id)
        else:
            self.exp_agent_formation = None
    
    def _dead_unit(self, agent, id):
        if not len(self.formations[id][0]):
            self.formations.pop(id)

    def _end_task(self, id):
        poss, action = self.in_misison[id].get()
        if action == "fight":
            self._end_fight(poss, id)
        elif action == "explore":
            self._end_explore(self.base, id)
        elif action == "go_to_formation" and id == -1:
            self.disjoin_explorer()
            self.def_explorer()
    
    def disjoin_explorer():
        pass

    def _end_explore(self, poss, id):
        pass

    def _end_fight(self, poss, id):
        self.attack_troop.pop(id)
        agents = self.formations[id][0]
        num_units = len(agents)
        if num_units != self.formations[id][1].N - 1:
            if num_units == 1:
                self.formations[id] = agents, OneFormation(poss)
            else:
                self.formations[id] = agents, TwoRowsFormation(num_units, poss)
        self.in_misison[id] = Queue()
        self.in_misison[id].put((poss, "go_to_formation"))
        self.troops[id] = MediumAgentMove(agents, self.formations[id][1], self.roadmap, self.low_events, id)
        self.troops[id].go_to_formation()

    def set_move(self, formation):
        pass

    def set_figth(self, id, troop, enemy):
        mision = self.in_misison.get(-1) if self.in_misison else None
        if mision not in self.in_misison:
            agent, formation = self.troops.pop(id).get_info()
            self.in_misison[id] = Queue()
            self.in_misison[id].put((formation.poss, "fight"))
            self.attack_troop[id] = MediumAgentFigth(agent, enemy, self.map, self.low_events, id)
            self.attack_troop[id].figth_in_formation()
            self.go_to_enemies(troop.formation.poss)
