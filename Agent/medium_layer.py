from IA.basicAlgorithms import RRAstar, RoadMap, whcastar_search
from entities.utils import DIRECTIONS
from IA.formations import Formation
from .low_layer import BasicAgent
from IA._definitions import *
from Agent.ia import *
import numpy as np
from math import ceil

class MediumAgentMove:
    def __init__(self, window_size,agents: set[BasicAgent], formation: Formation, roadmap: RoadMap, events: dict, ocupations, global_time, id) -> None:
        self.window_size = window_size
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
        self.ocupations = ocupations
        self.formados = False
        self.global_time: GlobalTime = global_time
        self.positioned: bool = None
        self.all_goals: list[bool] = None
        self.rrastar_list: list[RRAstar] = None
        self.from_to: list[tuple[int, int]] = None
        self.all_path: dict[BasicAgent, list[tuple[int, int]]] = None
        self.max_cost: int = max(map(lambda x: x.get_move_cost(), agents))
        self.time: int = self.max_cost
        self.low_events: dict = { 'is_invalid': self._is_invalid, 'end_task': self._end_task }
        self.path_index = -1

    def go_to_formation(self, poss = None, dirt = None):
        poss_from = self.formation.poss
        if dirt is not None:
            self.formation.rotate(dirt)
            self.dirt = dirt
        if poss is not None and self.formation.piss != poss:
            self.formation.set_in(*poss)

        start_poss = [u for u in self.agents]
        new_from_to =  self.formation.asign_formation(start_poss)
        start_poss.clear()
        go_to = []
        for agent, poss in new_from_to:
            start_poss.append(agent)
            go_to.append((agent.get_position(), poss))
        self.go_to_positions(poss_from, start_poss, go_to)

    def wait_positions(self):
        self.go_to_positions(self.formation.poss, self.agents, [(agent.get_position(), agent.get_position()) for agent in self.agents])

    def go_to_positions(self, poss_from, agents, goals):
        new_goal = False
        if self.from_to != None:
            for index, (before, after) in enumerate(zip(self.from_to, goals)):
                if before != after:
                    self.rrastar_list[index] = None
                    new_goal = True
        else:
            self.rrastar_list = np.ndarray(shape=len(agents), dtype=RRAstar)
            new_goal = True
        self.from_to = goals
        self.all_goals = list(map(lambda x: x[0].get_position() != x[1], zip(agents, goals)))
        self.positioned = not any(self.all_goals)

        ori_x, ori_y = self.formation.poss
        x, y = poss_from
        dx, dy = ori_x - x, ori_y - y

        if not len(self.invalidate) and (self.global_time.time % self.window_size == 0 or new_goal or len(self.available) == len(self.agents)):
            self.available.clear()
            agents, goals, self.rrastar_list = MediumAgentMove.order(agents, goals, self.rrastar_list)
            path = whcastar_search(agents, goals, self.rrastar_list, self.roadmap, self.window_size // self.max_cost, FakeReservation(self.ocupations, self.global_time.time, self.max_cost))
            path_list = list(path.items())

            for j in range(1, len(path_list[0][1])):
                for i in range(len(path_list)):
                    x, y = path_list[i][0].get_position()
                    x, y = x + dx, y + dy
                    (ax, ay), _ = path_list[i][1][j]

                    for k in range(self.max_cost):
                        self.ocupations[((ax, ay), self.global_time.time + j * self.max_cost + k)] = path_list[i][0]
            for agent in self.agents:
                path[agent].pop(0)
            self._notify_task(path.items())
        self._notify_move()

    def order(agents, goals, rrastar_list):
        newagents, newgoals, newarrastar_list = [], [], []
        zip_goals = list(zip(agents, goals, rrastar_list))
        zip_goals.sort(key=lambda x: x[0].get_position() == x[1])
        for agent, goal, rrastar in zip_goals:
            newagents.append(agent)
            newgoals.append(goal)
            newarrastar_list.append(rrastar)
        return newagents, newgoals, newarrastar_list

    def move_in_formation(self, poss = None):
        new_goal = False
        if type(self.ia) is not RoadMapMove_IA:
            new_goal = True
            self.ia = RoadMapMove_IA()
        if poss is not None and self.ia.goal != poss:
            self.path_index = 0
            self.all_path = []
            self.ia.set_goal(self.formation.poss, poss, self.roadmap)

        self._in_formation(new_goal)
    
    def explore_in_formation(self, area, size):
        new_goal = False
        if type(self.ia) is not SliceMapMove_IA:
            new_goal = True
            self.ia = SliceMapMove_IA()
        if area is not None and size is not None and self.ia.goal != (area, size):
            self.path_index = 0
            self.all_path = []
            self.ia.set_goal(self.formation.poss, (area, size), self.roadmap)

        self._in_formation(new_goal)

    def _in_formation(self, new_goal):
        if not len(self.invalidate) and (self.global_time.time % self.window_size == 0 or new_goal or len(self.available) == len(self.agents)):
            # self.path_index es el próximo indice a partir del cual tienes que volver a calcular
            path = dict([(agent, []) for agent in self.agents])

            self.get_real_formation_path(path, self.global_time.time, self.path_index, self.formados, self.window_size)
            if path:
                self.available.clear()
                self._notify_task(path.items())
            else: 
                return
        # TODO: marcar los ocupations de los invalidate 
        self._notify_move()

    def mark_ocupation(self, all_path, path, start_time, start_index = 0, window = np.infty):
        count = 0
        ori_x, ori_y = self.formation.poss
        direc = [(next_x - ori_x, next_y - ori_y) for next_x, next_y in all_path[start_index:]]
        for dx,dy in direc:
            end_windows = False
            for agent, x, y in [(x, *x.get_position()) for x in self.agents]:
                for i in range(self.max_cost):
                    poss = ((x + dx, y + dy), start_time + count * self.max_cost + i)
                    if poss in self.ocupations:
                        return start_index + count
                    if start_time + count * self.max_cost + i >= window + window * (self.global_time.time // window):
                        end_windows = True

            for agent, x, y in [(x, *x.get_position()) for x in self.agents]:
                if not end_windows: 
                    for i in range(self.max_cost):
                        poss = ((x + dx, y + dy), start_time + count * self.max_cost + i)
                        self.ocupations[poss] = agent
                path[agent].append(((x + dx, y + dy), start_time + count * self.max_cost))

            count += 1
            if end_windows:
                return start_index + count

        return start_index + count

    def check_in_ocupation(self, path, start_time, start_index, window):
        count = 0
        ori_x, ori_y = self.formation.poss
        direc = [(next_x - ori_x, next_y - ori_y) for next_x, next_y in path]
        for dx,dy in direc[start_index:]:
            ocupated = False
            end_windows = False
            for i in range(self.max_cost):
                if start_time + count * self.max_cost + i >= window + window * (self.global_time.time // window):
                    end_windows = True
                    break
                for x,y in [i.position for i in self.formation.nodes if i.state != self.formation.nodes[self.formation.main].state]:
                    poss = ((x + dx, y + dy), start_time + count * self.max_cost + i)
                    if self.ocupations.get(poss) is not None:
                        ocupated = True
                        break
                if ocupated:
                    break
    
            if end_windows:
                return start_index + count + 1
            if not ocupated:
                return start_index + count

            count += 1
        return start_index + count
    
    def get_real_formation_path(self, path, time, index, formados, window):
        # index es el primer indice a analizar
        current_time = time
        if not len(self.all_path):
            self.all_path = self.ia.get_move_for(self.formation.poss)
        if time % window == 0:
            current_time += self.max_cost
        last_valid_index = -1

        if(formados):
            while True:
                #en caso de final de ventana no marca el último a pesar de ser válido
                last_valid_index = self.mark_ocupation(self.all_path, path, current_time, index, window)
                #current time marca el tiempo que te demoras en recorrer todo lo valido
                current_time = current_time + self.max_cost * ((last_valid_index - index))
                if current_time + (window - (self.global_time.time % window)) % self.max_cost >= window + window * (self.global_time.time // window):
                    for i in range((window - (self.global_time.time % window)) % self.max_cost):
                        for agent, agent_path in path.items():
                            poss, t = agent_path[-1]
                            self.ocupations[(poss, t + i)] = agent
                    self.formados = True
                    self.path_index = last_valid_index
                    return

                index = last_valid_index

                if last_valid_index != len(self.all_path):
                    break

                new_poss = self.all_path[-1]
                new_path = self.ia.get_move_for(new_poss)
                if new_path is None:
                    self.formados = True
                    self.path_index = last_valid_index
                    return
                self.all_path += new_path
        else:
            # Cuando no esta formado index no es valido
            last_valid_index = index
        
        first_time = current_time - self.max_cost
        first_next_valid = 0
        goal_index = -1
        while True:
            first_next_valid = self.check_in_ocupation(self.all_path, current_time, index, window)
            # marca el tiempo que te demoras en recorrer los invalidos
            current_time = current_time + self.max_cost * (first_next_valid - index)

            if current_time + (window - (self.global_time.time % window)) % self.max_cost >= window + window * (self.global_time.time // window):
                current_time = window + window * (self.global_time.time // window)
                # Siempre que puedes coges de goal la posicion en all_path despues de la ventana
                if first_next_valid == len(self.all_path):
                    goal = self.all_path[first_next_valid - 1]
                    goal_index = first_next_valid - 1
                    break
                goal = self.all_path[first_next_valid]
                goal_index = first_next_valid
                break
            
            if first_next_valid != len(self.all_path):
                goal = self.all_path[first_next_valid]
                goal_index = first_next_valid
                break


            #el index se usa en la proxima iteracion del while y tiene que ser el primer indice 
            index = len(self.all_path) - 1
            new_path = self.ia.get_move_for(self.formation.poss, self.ocupations)
            if new_path is None:
                goal = self.all_path[first_next_valid]
                break 
            self.all_path += new_path
        # window + window * (self.global_time.time // window) final de la ventana
        fix_windows = min(current_time - first_time + 1, window + window * (self.global_time.time // window) - first_time)
        # fix_windows = current_time - first_time + 1
        
        start = [agent_path[-1][0] for agent_path in path.values()] if len(path.values().__iter__().__next__()) else [agent.get_position() for agent in self.agents]
        formados = self.fix_path(path, start, goal, first_time, fix_windows)
        if(formados):
            if(path.values().__iter__().__next__()[-1][1] + (window - (self.global_time.time % window)) % self.max_cost >= window + window * (self.global_time.time // window)):
                for i in range((window - (self.global_time.time % window)) % self.max_cost):
                    for agent, agent_path in path.items():
                        poss, t = agent_path[-1]
                        self.ocupations[(poss, t + i)] = agent
                self.formados = True
                # self.path_index cumple lo mismo que el index del llamado recursivo
                # es el proximo indice a analizar
                # se asume que si logro formarse en el whca* entonces se formo en first_next_valid
                self.path_index = goal_index + 1
                return
            self.get_real_formation_path(path, path.values().__iter__().__next__()[-1][1] + self.max_cost, goal_index + 1, True, window)
        else:
            x_sum = 0
            y_sum = 0
            count = 0
            for path_by_agent in path.values():
                (last_x, last_y), _ = path_by_agent[-1]
                x_sum += last_x
                y_sum += last_y
                count += 1
            
            cluster = (ceil(x_sum/count),ceil(y_sum/count))

            next_index = 0
            best_value = np.infty
            for i in range(last_valid_index, first_next_valid + 1):
                if(best_value > norma_inf(self.all_path[i], cluster)):
                    next_index = i
                    best_value = norma_inf(self.all_path[i], cluster)

            if(path.values().__iter__().__next__()[-1][1] + (window - (self.global_time.time % window)) % self.max_cost >= window + window * (self.global_time.time // window)):
                self.formados = False
                self.path_index = next_index + 1
                return


    def fix_path(self, path, start, goal, first_time, fix_windows):
        ori_x, ori_y = self.formation.poss
        dx, dy = goal[0] - ori_x, goal[1] - ori_y
        goals = list(zip(
            start, [(i.position[0] + dx, i.position[1] + dy) for i in self.formation.nodes if i.state != self.formation.nodes[self.formation.main].state])
        )

        ids = []
        rrastar = []
        for agent in self.agents:
            ids.append(agent)
            rrastar.append(None)

        path_for_agent = list(whcastar_search(ids, goals, rrastar, self.roadmap, fix_windows // self.max_cost, FakeReservation(self.ocupations, first_time, self.max_cost)).items())

        path_for_agent.sort(key = lambda x: x[1][0][0])
        path_for_agent.sort(key = lambda x: x[1][0][1])
        goals.sort(key = lambda x: x[0][0])   
        goals.sort(key = lambda x: x[0][1])  
        for j in range(1, len(path_for_agent[0][1])):
            all_right = True
            for i in range(len(path_for_agent)):
                x, y = goals[i][1]
                (ax, ay), _ = path_for_agent[i][1][j]

                for k in range(self.max_cost):
                    self.ocupations[((ax, ay), first_time + j * self.max_cost + k)] = path_for_agent[i][0]
                path[path_for_agent[i][0]].append(((ax, ay), first_time + j * self.max_cost))

                if ax != x or ay != y:
                    all_right = False

            if all_right:
                return True
        
        return False
        
    def get_info(self):
        return self.agents, self.formation

    def inform_view(self, view):
        oponents = set()
        for ag in self.agents:
            oponents = oponents.union(view(state = ag.get_position(), vision = ag.connector.unit.get_vision_radio))
        return list(oponents)

    def _notify_formation_task(self, path: list[tuple]):
        x, y = self.formation.poss
        direc = [(next_x - x, next_y - y) for next_x, next_y in path]
        for unit in self.agents:
            actions = []
            x, y = unit.get_position()
            prev_move = unit.get_position()
            for dx, dy in direc:
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
            self.formation.set_in(*prev_move)
            unit.set_action_list(actions, self.max_cost - unit.get_move_cost(), self.low_events)

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

    def __init__(self, agents, oponents, map,  events: dict, ocuped_table, global_time, id):

        self.id = id
        self.events = events
        self.agents = agents
        self.oponents = oponents or []
        self.alliads = []
        self.strategie = GroupFigthGame()
        self.available = agents.copy()
        self.map = map
        self.ocuped_table = ocuped_table
        self.global_time = global_time
        self.team = list(agents)[0].connector.agent.id 
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
                actual = norma2(ag.get_position(), oponent.get_position())
                if actual < nearest:
                    nearest = actual
                    best = oponent
            result.append((ag,best))
        return result 
            
    def view(self, view):
        neighbors = set()
        for ag in self.agents:
            neighbors = neighbors.union(view(state = ag.get_position(),vision = ag.connector.unit.get_vision_radio) or []) 
        self.oponents, self.alliads = self.split_neighbors(neighbors, self.oponents,self.alliads)
    
    def split_neighbors(self, neighbors, oponents, alliads):
        oponents =list(filter(lambda x: x.is_connected() , set([self.map[i].unit for i in filter(lambda x: not self.map[x].is_empty and self.map[x].unit.agent.id != self.team, neighbors)]).union(oponents)))
        alliads = [self.map[i].unit for i in filter(lambda x: self.map[x].unit.agent.id == self.team, neighbors)]

        return oponents, alliads

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
    
    # def figth_in_formation(self):
    #     #Se asignan las parejas de lucha segun distancia entre los enemigos y los conectores que estan sin hacer nada
        
    #     recalc = self.asign_figters(self.available, self.oponents)
    #     bussy = {} 
    #     best_move = {}
    #     attacks = []
    #     wait = []
    #     while recalc: #mientras que haya parajas a las que recalcularle el minimax
    #         x, y = recalc.pop(0)
    #         state = State(self.map, x.connector, self.map[y].unit, bussy.get(x.connector.id) or []) #se crea un estado que contemple el estado actual de la pareja
    #         #en combate, el mapa y las ubicaciones en las que ya no tiene sentido que se paren
            
    #         value, action = h_alphabeta_search_solution(self.strategie, state, lambda x, y, z: z >= 3, self.strategie.heuristic)
            
    #         if action[0] == "wait":
    #             wait.append((x, action[1]))
    #         elif action[0] == "move": #si la accion es un move entonces hay que ver si no confluyen variosmove a una misma casilla
    #             bus =  bussy.get(x.connector.id) or []
    #             bus.append(action[1])
    #             bussy[x.connector.id] = bus

    #             move = best_move.get(action[1])
    #             if move is not None: #en caso que varios se muevan al mismo lugar se coge el de resultado mas prometedor
    #                 x0,y0,V = move
    #                 if V < value:
    #                     move = (x,y,value)
    #                     recalc.append((x0,y0))
    #                 elif V >= value:
    #                     recalc.append((x,y))
    #             else: move = (x,y,value)
    #             best_move[action[1]] = move
    #         elif action[0] == "attack":
    #             attacks.append((x,action[1])) #si es un ataque siempre se puede ejecutar 
    #     self.set_actions(best_move, attacks, wait)
    #     self.eject_actions()
    
    def figth_in_formation(self):
        bussy = {}
        free = []
        best_move = {}
        gen_bussy = []
        actions = []
        attack = 0
        alliadslen = len(self.alliads)
        # for agent, y  in recalc:
        # for agent in self.available:
        while len(self.available):
            agent = self.available.pop()
            self.alliads.remove(agent.connector)

            if agent.get_move_cost() == inf:
                actions.append((agent,self.static_action(agent)))
            else:

                # state = State(self.map, agent.connector, y, bussy) #se crea un estado que contemple el estado actual de la pareja
                # value, state = h_alphabeta_search_solution(self.strategie, state, lambda x, y, z: z >= 3, self.strategie.heuristic)
                current_busy = []
                current_busy+=bussy.get(agent.connector.id) or []
                current_busy +=  gen_bussy
                state = Group_State(self.map, agent.connector, self.alliads,self.oponents, attack,alliadslen,self.global_time.time, current_busy, self.ocuped_table)
                value, state = h_alphabeta_search_solution(self.strategie,state,lambda x, y, z: z >= 2,self.strategie.heuristic)
                
                if state is not None:
                    action, pos = state
                    if action == "attack":
                        attack =+ agent.connector.unit.get_damage
                        gen_bussy.append(agent.get_position())
                        for i in range(agent.get_move_cost() +1):
                            self.ocuped_table[((agent.get_position()), self.global_time.time + i)] = agent
                        actions.append((agent,("attack", pos)))
                    if action == "wait":
                        self.ocuped_table[((agent.get_position()), self.global_time.time)] = agent
                        gen_bussy.append(agent.get_position())
                        actions.append((agent,("wait", pos)))
                    if action== "move":
                       
                        bus =  bussy.get(agent.connector.id) or []
                        bus.append(pos)
                        bussy[agent.connector.id] = bus
  
                        move = best_move.get(pos)
                        if move is not None: #en caso que varios se muevan al mismo lugar se coge el de resultado mas prometedor
                            x, V = move
                            if V < value:
                                move = (agent,value)
                                self.available.add(x)
                                self.alliads.append(x.connector)
                            elif V >= value:
                                self.available.add(agent)
                                self.alliads.append(agent.connector)

                        else: move = (agent,value)
                        best_move[pos] = move
        for poss, (agent,_) in best_move.items():
            for i in range(agent.get_move_cost() +1):
                self.ocuped_table[((poss), self.global_time.time + i)] = agent
                       
            actions.append((agent,("move", poss)))
                          
        self.set_actions(actions)
        self.eject_actions()

    def static_action(self, agent):
        best = infinity
        pos = None
        range = agent.connector.unit.get_attack_range
        position = agent.get_position()
        for enemy in self.oponents:
            if norma_inf(position, enemy.get_position()) <= range and  enemy.get_health_points() < best:
                best = enemy.get_health_points()
                pos = enemy.get_position()
        
        return ("wait", position) if pos is None else ("attack", pos)

    def set_actions(self, actions):
        for x , (action, pos) in actions:
            x.set_action_list([(action, pos)], events = self.my_events, rithm= 0, change_prev = True)

              
    def eject_actions(self):
        for agent in self.agents:
            agent.eject_action()   
    
    def inform_figth():
        pass  
    
    def inform_view():
        pass
    