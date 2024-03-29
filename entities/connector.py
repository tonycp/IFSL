from math import sqrt
from random import randint

from matplotlib.pyplot import connect
from ._units import *
from .utils import STATES, direction_to_int, I_DIR, J_DIR, int_to_direction, validMove
from IA.basicAlgorithms import *
from IA._definitions import Node

def discrete_distance(x_0,y_0,x_1,y_1):
    return sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)

class StateMannager:
    class Connector:
        __id = 0
        def __init__(self,agent, unit:Unit, position, functions) -> None:
            self.agent = agent
            self.unit = unit 
            self.__current_hp = unit.get_health_points
            self.__current_position = position
            StateMannager.Connector.__id += 1
            self.state = STATES.Stand
            self.prev_dir = None
            self.prev_connector_to_swap = None
            self.timer = 0
            self.time_to_wait = 0
            self.id = StateMannager.Connector.__id
            for func in functions:
                self.__dict__[func.__name__] = func

        def update_position(self,new_position):
            self.__current_position = new_position

        def update_health_points(self, new_hp):
            if self.is_connected():
                self.__current_hp = new_hp
                return True
            return False

        def get_position(self):
            return self.__current_position

        def get_health_points(self):
            return self.__current_hp

        def is_connected(self):
            return self.__current_hp > 0 

        def __eq__(self, __o: object) -> bool:
            return __o is not None and __o.id == self.id
        
        def __hash__(self) -> int:
            return self.id
        def __repr__(self) -> str:
            return str(self.id)
    
    def __init__(self, map, agents_units_positions):
        self.map = map
        self.roadmap = RoadMap(map, Knight())
        self.not_move_log = []
        self.agents = []
        self.move_notifications = []
        self.swap_notifications = []
        self.attack_notifications = []
        self.current_log = []
        for ag, formations in agents_units_positions:
            self.agents.append(ag)
            for units, formation in formations:
                connectors = []
                for unit, position in units:
                    connector = self.create_connector(ag,unit,position)
                    map[position].unit = connector
                    connectors.append(connector)
                ag.connect(connectors, formation)
            


    def create_connector(self,agent,unit:Unit,position):
        return StateMannager.Connector(agent,unit,position,[self.notify_move, self.notify_not_move, self.notify_swap, self.notify_not_swap, self.notify_attack])

    def exec_round(self):
        # Pedir acciones a cada una de las IAs
        
        for agent in self.agents:
            problem = ViewProblem(self.roadmap)
            filter_enemy = lambda x: not self.map[x.state].is_empty and self.map[x.state].unit.agent.id != agent.id
            filter_all = lambda x: not self.map[x.state].is_empty 

            adj = lambda x: expand(problem, x)
            agent.decide(lambda state, vision, filter: list(map(lambda x: x.state, breadth_first_search(NodeTree(state=state), filter, adj, None, vision))), filter_enemy, filter_all)
            
        # Ejecutar las acciones
        log = self.__exec_actions()
        return log

    def __exec_actions(self):
        log = []
        self.exec_move(log,self.not_move_log)
        self.exec_swap(log)
        self.exec_attack(log)
        #Pasando los movimientos que fueron invalidados
        ia_list = []
        connectors_by_ia = []
        for connector, num_dir in  self.not_move_log:
            index = -1
            for i in range(len(ia_list)):
                if(ia_list[i] == connector.agent):
                    index = i
                    break
            if index == -1:
                ia_list.append(connector.agent)
                connectors_by_ia.append([(connector,int_to_direction(num_dir))])
            else:
                connectors_by_ia[index].append((connector,int_to_direction(num_dir)))
        for i in range(len(ia_list)):
            ia_list[i].invalidations(connectors_by_ia[i])
        
        self.move_notifications = []
        self.attack_notifications = []
        self.swap_notifications = []
        self.not_move_log = []
        return log

    def not_move_subtree(self,node, not_move_log):
        connector, num_dir, _ = node.state
        self.map[connector.get_position()].unit = connector
        not_move_log.append((connector,num_dir))
        for child in node.actions:
            self.not_move_subtree(child,not_move_log)
    
    def move_connector(self,connector,num_dir, move_log, delete_prev = True):
        i, j = connector.get_position()
        move_log.append(self.map[(i,j)])
        new_i = i + I_DIR[num_dir]
        new_j = j + J_DIR[num_dir]
        move_log.append(self.map[(new_i,new_j)])
        if delete_prev:
            self.map[i,j].unit = None
        self.map[new_i,new_j].unit = connector
        connector.update_position((new_i,new_j))
        return new_i, new_j

    def resolve_move_dependencies(self,node,not_move_log, move_log):
        if(not node.actions):
            return
        if(len(node.actions) == 1):
            self.move_connector(node.actions[0].state[0],node.actions[0].state[1],move_log)
            self.resolve_move_dependencies(node.actions[0],not_move_log,move_log)
            return
        
        total = sum([child.state[0].unit.get_move_cost for child in node.actions])
        ran = randint(1,total)
        founded = False
        for child in node.actions:
            ran -= child.state[0].unit.get_move_cost
            if(not founded and ran <= 0):
                self.move_connector(child.state[0],child.state[1],move_log)
                self.resolve_move_dependencies(child,not_move_log,move_log)
                founded = True
            else:
                self.not_move_subtree(child,not_move_log)

    def resolve_cycle(self,node: Node,move_log:list):
        if(node.state[2] == -1):
            return
        self.move_connector(node.actions[0].state[0], node.actions[0].state[1], move_log, False)
        connector, num_dir,_ = node.state
        node.state = (connector, num_dir, -1)
        self.resolve_cycle(node.actions[0], move_log)

    def exec_move(self,move_log,not_move_log):
        resolve_roots = []
        moved_nodes = []
        node_states = []
        resolve_cycles = []
        #Actualizando los timers
        for index in range(len(self.move_notifications)):
            connector, num_dir = self.move_notifications[index]
            if(not connector.is_connected):
                continue
            if(connector.prev_dir is None or connector.prev_dir != num_dir or connector.state != STATES.Moving):
                connector.timer = connector.unit.get_move_cost
                connector.prev_dir = num_dir
                connector.state = STATES.Moving
            connector.timer -= 1
            if(connector.timer == 0): # Marcando los que se tienen que mover
                self.map[connector.get_position()].unit = Node((connector,num_dir,-1),[])
                moved_nodes.append(index)
        #Añadiendo aristas y podando nodos
        for index in moved_nodes:
            connector, num_dir = self.move_notifications[index]
            i, j = connector.get_position()
            if type(self.map[i,j].unit) is StateMannager.Connector or self.map[i,j].unit.state[2] != -1:
                continue
            self.map[i,j].unit.state = (connector,num_dir,len(node_states))
            node_states.append(NodeState())
            while True:
                new_i = i + I_DIR[num_dir]
                new_j = j + J_DIR[num_dir]
                if self.map[new_i,new_j].is_empty:
                    self.map[new_i,new_j].unit = Node("Root",[(self.map[i,j].unit)])
                    node_states[self.map[i,j].unit.state[2]].visited = True
                    resolve_roots.append(self.map[new_i,new_j].unit)
                    break
                else:
                    current_node = self.map[i,j].unit
                    next_node = self.map[new_i,new_j].unit
                    # Si se mueve a un nodo que no se mueve en ese turno
                    if(not (type(next_node) is Node)):
                        self.not_move_subtree(current_node,not_move_log)
                        break
                    if(next_node.state == "Root"):
                        node_states[current_node.state[2]].visited = True
                        next_node.actions.append(current_node)
                        break
                    # Si se mueve en direccion contraria a la que se mueve otro adyacente
                    if(abs(next_node.state[0].prev_dir - current_node.state[0].prev_dir) == 4):
                        self.not_move_subtree(current_node,not_move_log)
                        self.not_move_subtree(next_node,not_move_log)
                        break

                    elif(next_node.state[2] == -1):
                        connector, num_dir,_ = next_node.state
                        next_node.state = (connector, num_dir,current_node.state[2])
                        i, j = next_node.state[0].get_position()
                        num_dir = next_node.state[1]
                        next_node.actions.append(current_node)
                        continue
                    elif(node_states[next_node.state[2]].visited):
                        #
                        if(node_states[next_node.state[2]].is_cycle):
                            self.not_move_subtree(current_node,not_move_log)
                            break
                        else:
                            node_states[current_node.state[2]].visited = True
                            next_node.actions.append(current_node)
                            break
                    else:
                        if(next_node.actions):
                            self.not_move_subtree(next_node.actions[0])
                            next_node.actions = []
                        next_node.actions.append(current_node)
                        node_states[current_node.state[2]].visited = True
                        node_states[current_node.state[2]].is_cycle = True
                        resolve_cycles.append(current_node)
                        break
                        #self.map[new_i,new_j].unit.actions.append(self.map[i,j].unit)
        
        #Resolver arbol de dependencias
        for root in  resolve_roots:
            self.resolve_move_dependencies(root,not_move_log,move_log)

        for cycle in resolve_cycles:
            self.resolve_cycle(cycle,move_log)

        for end_mov_connector in moved_nodes:
            node = self.move_notifications[end_mov_connector][0]
            node.state = STATES.Stand
            connector.prev_dir = None
            node.timer = 0
        self.move_notifications = []
    
    def exec_swap(self, move_log):
        for connector,other_connector, num_dir in self.swap_notifications:
            #Si el otro participante del swap no está en Stand cancelar el swap
            if(other_connector.state != STATES.Stand):
                self.notify_not_swap(connector)
                continue
            #Si está iniciando el swap o cambió de dirección o objetivo
            if(not connector.prev_dir or not connector.prev_connector_to_swap or connector.state != STATES.Swaping
            or connector.prev_dir != num_dir or (connector.prev_connector_to_swap and connector.prev_connector_to_swap != other_connector)):
                connector.timer = max(connector.unit.get_move_cost,other_connector.unit.get_move_cost)
                connector.prev_connector_to_swap = other_connector
                connector.state = STATES.Stand
                connector.prev_dir = num_dir
            connector.timer -= 1
            if(connector.timer == 1):
                self.map[connector.get_position()].unit = other_connector
                self.map[other_connector.get_position()].unit = connector
                original_pos = connector.get_position()
                connector.update_position(original_pos)
                other_connector.update_position(connector)
                move_log.append(self.map[connector.get_position()])
                move_log.append(self.map[other_connector.get_position()])
    
    def exec_attack(self, log):
        for connector, pos_i, pos_j in self.attack_notifications:
            other_connector = self.map[pos_i, pos_j].unit
            if(other_connector):
                was_alive = other_connector.is_connected()
                if(other_connector.state == STATES.Stand):
                    other_connector.update_health_points(other_connector.get_health_points() - connector.unit.get_damage)
                    log.append(self.map[other_connector.get_position()])
                else:
                    if(connector.unit.get_move_cost == inf or randint(1,other_connector.unit.get_move_cost + connector.unit.get_move_cost) <= other_connector.unit.get_move_cost):
                        other_connector.update_health_points(other_connector.get_health_points() - connector.unit.get_damage)
                        log.append(self.map[other_connector.get_position()])

                # if(was_alive and not other_connector.is_connected) or (not was_alive and other_connector.is_connected):
                if(was_alive and not other_connector.is_connected()):
                    other_connector.agent.disconnect(other_connector)
                    position = other_connector.get_position()
                    self.map[position].unit = None
                    log.append(self.map[position])


    def is_valid_move(self,connector: Connector,num_dir : int):
        i, j = connector.get_position()
        new_i = i + I_DIR[num_dir]
        new_j = j + J_DIR[num_dir]
        height, width = self.map.shape
        if not validMove(new_i, new_j, height, width): return False
        cell_to_move = self.map[new_i,new_j]
        return cell_to_move.crossable(connector.unit)
    
    def notify_move(self, connector : Connector, direction):
        num_dir = direction_to_int(direction)
        if self.is_valid_move(connector, num_dir):
            self.move_notifications.append((connector, num_dir))
        else:
            self.not_move_log.append((connector, num_dir))


    def notify_not_move(self,connector : Connector):
        connector.state = STATES.Stand
        connector.prev_dir = None
        connector.timer = 0
        for i in range(len(self.move_notifications)):
            if(self.move_notifications[i][0] == connector):
                self.move_notifications.pop(i)
                break

    def notify_swap(self,connector,direction):
        num_direction = direction_to_int(direction)
        i, j = connector.get_position()
        new_i = i + I_DIR[num_direction]
        new_j = j + J_DIR[num_direction]
        height, width = self.map.shape
        if(new_i < height and new_j < width):
            other_connector = self.map[i + I_DIR[num_direction] , j + J_DIR[num_direction]].get_connector()
            if( other_connector and other_connector.agent == connector.agent):
                self.swap_notifications.append((connector,other_connector,num_direction))
                return
            

    def notify_not_swap(self,connector:Connector):
        connector.timer = 0
        connector.state = STATES.Stand
        connector.prev_connector_to_swap = None
        connector.prev_dir = None
        
    def notify_attack(self, connector:Connector, pos_i, pos_j):
        i, j = connector.get_position()
        height, width = self.map.shape
        if(pos_i < height and pos_j < width and norma_inf((i, j), (pos_i, pos_j)) <= connector.unit.get_attack_range):
            self.attack_notifications.append((connector, pos_i, pos_j))


    
class NodeState:
    def __init__(self,visited = False, is_cycle = False):
        self.visited = visited
        self.is_cycle = is_cycle
        
     

    