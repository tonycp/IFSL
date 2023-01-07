from math import inf
from IA.formations import Formation, OneFormation, TwoRowsFormation
from Agent.medium_layer import MediumAgentMove, MediumAgentFigth
from IA.basicAlgorithms import RoadMap
from IA._definitions import GlobalTime
from random import shuffle

class HighAgent:    
    percent_explorer = 0.25
    def __init__(self, map, roadmap: RoadMap, formations: dict[int, tuple[set, Formation]], base ,window_size = 15):
        self.map = map
        self.window_size = window_size
        self.roadmap = roadmap
        self.formations = formations
        self.explorer = []
        self.troops = {}
        self.in_misison:dict[int, list]={}
        self.global_time = GlobalTime()
        self.ocupations = {}
        self.base = base
        self.exp_formation = None
        self.exp_agent_formation = None
        self.low_events = { "dead_formation": self._dead_formation, "dead_unit": self._dead_unit, "end_task": self._end_task }

        for id, (agents, formation) in self.formations.items():
            self.troops[id] = MediumAgentMove(self.window_size, agents, formation, self.roadmap, self.low_events, self.ocupations, self.global_time, id)
            self.in_misison[id] = []
        self.def_explorer()

    def decide(self, view, filter1, filter2):
        if self.global_time.time % self.window_size == 0:
            for id, (agents, formation) in self.formations.items(): 
                troop = self.troops[id]
                if(type(troop) is MediumAgentFigth): continue
                for i in range(min(troop.max_cost, self.window_size)):
                    for agent in agents:
                        self.ocupations[(agent.get_position(), i + self.global_time.time)] = agent

        if self.troops.get(1):
            a_agent = list(self.troops[1].agents)[0]
            if a_agent.connector.agent.id == 1:
                print(f"{a_agent.connector.id}, {a_agent.get_position()}, {a_agent.prev}, {a_agent.connector.timer}, {self.global_time.time}")

        if self.troops.get(2):
            a_agent = list(self.troops[2].agents)[0]
            if a_agent.connector.agent.id == 1:
                print(f"{a_agent.connector.id}, {a_agent.get_position()}, {a_agent.prev}, {a_agent.connector.timer}, {self.global_time.time}")

        for id, troop in list(self.troops.items()):
            if type(troop) is MediumAgentFigth:
                continue
            troop_view = troop.inform_view(lambda state, vision : view(state =state, vision = vision, filter = filter1))
            if len(troop_view) > 0:
                self.set_figth(id, troop, troop_view)

        for id, move in self.in_misison.items():
            poss, action = move[0] if len(move) else (None, None)
            if action == "fight":
                if type(self.troops[id]) is not MediumAgentFigth:
                    agents = self.troops[id].agents
                    self.troops[id] = MediumAgentFigth(agents, None, self.map, self.low_events, self.ocupations, self.global_time, id)
                self.troops[id].view(lambda state, vision : view(state =state, vision = vision, filter = filter2))
                self.troops[id].figth_in_formation()
            elif action:
                if type(self.troops[id]) is not MediumAgentMove:
                    formation = self.formations[id][1]
                    agents = self.troops[id].agents
                    self.troops[id] = MediumAgentMove(self.window_size,agents, formation, self.roadmap, self.low_events, self.ocupations, self.global_time, id)

                if action == "move_in_formation":
                    self.troops[id].move_in_formation(poss)
                elif action == "go_to_formation":
                    self.troops[id].go_to_formation()
                elif action == "explore":
                    self.troops[id].explore_in_formation(*poss)
        
        for id, troop in self.troops.items():
            if len(self.in_misison[id]):
                continue
            troop.wait_positions()

        self.global_time.tick()

    def go_to_enemies(self, id, positions):
        for key, agent in self.troops.items():
            mision = self.in_misison[key]
            if key != id and (positions, "fight") not in mision:
                self.in_misison[key].insert(0, (positions, "fight"))
                self.in_misison[key].insert(0, (positions, "move_in_formation"))
                self.clear_ocupate(agent)

    def def_explorer(self):
        bussy = set()
        explorer = list(filter(lambda x: len(x[1][0]) == 1, self.formations.items()))
        explorer.sort(key=lambda x: list(x[1][0])[0].get_move_cost())

        if len(explorer) == 0:
            return

        # mejorar los bussy
        for agent in self.troops.values():
            bussy = bussy.union(self.roadmap.color[agent.formation.poss])

        explorer_len = min(len(explorer), self.roadmap.color_num - len(bussy)) # arreglar
        self.explorer = explorer[:explorer_len]
        asing_num = list(map(lambda x: x[0], self.explorer))

        for num in asing_num:
            self.in_misison[num] = [([set(), None], "explore")]

        count = 0
        for area, size in self.roadmap.size.items():
            if area in bussy:
                continue
            index = count % explorer_len
            if index == 0:
                shuffle(asing_num)
            asing = self.in_misison[asing_num[index]][0][0]
            asing[0].add(area)
            asing[1] = HighAgent.update_size(asing[1], size)
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
        self.in_misison.pop(id)
        self.troops.pop(id)
    
    def _dead_unit(self, agent, id):
        if not len(self.formations[id][0]):
            self.formations.pop(id)

    def _end_task(self, id):
        poss, action = self.in_misison[id].pop(0)
        if action == "fight":
            self._end_fight(poss, id)
        elif action == "explore":
            self._end_explore(poss, id)
    
    def disjoin_explorer():
        pass

    def _end_fight(self, poss, id):
        agents = self.formations[id][0]

        if id == -1:
            self.disjoin_explorer()
            self.def_explorer()
        else:
            num_units = len(agents)
            if num_units != self.formations[id][1].N - 1:
                if num_units == 1:
                    self.formations[id] = agents, OneFormation(poss)
                else:
                    self.formations[id] = agents, TwoRowsFormation(num_units, poss)
            self.troops[id] = MediumAgentMove(self.window_size,agents, self.formations[id][1], self.roadmap, self.low_events, self.ocupations, self.global_time, id)
            self.in_misison[id].insert(0, (poss, "go_to_formation"))

    def _end_explore(self, poss, id):
        self.in_misison[id].append(poss, "explore")

    def set_figth(self, id, troop, enemy):
        px, py = 0, 0
        for x, y in enemy:
            px += x
            py += y
        px //= len(enemy)
        py //= len(enemy)
        self.in_misison[id] = [((px, py), "fight")]
        self.clear_ocupate(troop)
        self.go_to_enemies(id, (px, py))
    
    def clear_ocupate(self, troop):
        for agent in troop.agents:
            if agent.get_move_cost() == inf: continue
            out_index = -1
            for index, (_, poss) in enumerate(agent.action_list):
                if index == len(agent.action_list) - 1:
                    out_index = index
                    break
                # corr = agent.current_time if agent.current_time >= 0 else troop.max_cost
                num = self.global_time.time + agent.connector.timer + (agent.rithm - agent.current_time) + (index + 1) * troop.max_cost
                for time in range(troop.max_cost):
                        self.ocupations.pop((poss, num + time))
            if out_index != -1:
                num = self.global_time.time + agent.connector.timer + (agent.rithm - agent.current_time) + (out_index + 1) * troop.max_cost
                end = self.global_time.time + troop.max_cost - (self.global_time.time + 1) % troop.max_cost + out_index * troop.max_cost
                windows = self.window_size + self.window_size * (self.global_time.time // self.window_size)
                for i in range(windows - end):
                        self.ocupations.pop((poss, num + i))