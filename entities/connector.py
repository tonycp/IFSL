from cmath import sqrt
from random import randint
from numpy import matrix
from ._units import Unit
from .utils import STATES, direction_to_int, I_DIR, J_DIR, int_to_direction, validMove


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
            self.__id = StateMannager.Connector.__id
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
            return __o.__id == self.__id
    
    def __init__(self, map, agents_units_positions):
        self.map : matrix = map
        self.agents = []
        self.move_notifications = []
        self.swap_notifications = []
        self.attack_notifications = []
        self.current_log = []
        for ag, units in agents_units_positions:
            self.agents.append(ag)
            for unit,position in units:
                connector = self.create_connector(ag,unit,position)
                ag.connect(connector)
                map[position].set_unit(connector)


    def create_connector(self,agent,unit:Unit,position):
        return StateMannager.Connector(agent,unit,position,[self.notify_move, self.notify_not_move, self.notify_swap, self.notify_not_swap, self.notify_attack])

    def exec_round(self):
        # Pedir acciones a cada una de las IAs
        for agent in self.agents:
            # Calcular vision para este agente
            agent.decide()
        # Ejecutar las acciones
        log = self.__exec_actions()
        return log

    def __exec_actions(self):
        log = []
        self.exec_move(log)
        self.exec_swap(log)
        self.exec_attack(log)
        self.move_notifications = []
        self.attack_notifications = []
        self.swap_notifications = []
        return log

    def exec_move(self,log):
        collisions = dict()
        end_movements = []
        for connector, num_dir in self.move_notifications:
            if(not connector.prev_dir or direction_to_int(connector.prev_dir) != num_dir or connector.state != STATES.Moving):
                connector.timer = connector.unit.get_move_cost
                connector.prev_dir = int_to_direction(num_dir)
                connector.state = STATES.Moving
            connector.timer -= 1
            if(connector.timer == 0):
                i, j = connector.get_position()
                new_i = i + I_DIR[num_dir]
                new_j = j + J_DIR[num_dir]
                end_movements.append(connector)
                if self.map[new_i,new_j].is_empty:
                    self.map[i,j].set_unit()
                    self.map[new_i,new_j].set_unit(connector)
                    log.append(self.map[i,j])
                    log.append(self.map[new_i,new_j])
                    connector.update_position((new_i,new_j))
                else:
                    arr = collisions.get((new_i,new_j))
                    if arr:
                        arr.append(connector)
                    else:
                        collisions[(new_i,new_j)] = [self.map[new_i,new_j].get_unit,connector]

        for pos, connector_list in collisions.items:
            ran = randint(1,sum([con.unit.get_move_cost for con in connector_list])) 
            for con in connector_list:
                ran -= con.unit.get_move_cost
                if ran <= 0:
                    if(con.get_position() != pos):
                        #poner al que se había usado para marcar a su posicion original
                        old_con = self.map[pos].get_unit
                        self.map[pos].set_unit()
                        old_i = pos[0] + I_DIR[(direction_to_int(old_con.prev_dir) + 4)%4]
                        old_j = pos[1] + J_DIR[(direction_to_int(old_con.prev_dir) + 4)%4]
                        self.map[(old_i,old_j)].set_unit(old_con)
                        #poner al que es en su posicion correcta
                        self.map[con.get_position()].set_unit()
                        self.map[pos].set_unit(connector)
                        log.append(self.map[i,j])
                        log.append(self.map[new_i,new_j])
                        connector.update_position((new_i,new_j))
                    break


        for end_mov_connector in end_movements:
            end_mov_connector.state = STATES.Stand
            connector.prev_dir = None
    
    def exec_swap(self, log):
        for connector,other_connector, num_dir in self.swap_notifications:
            i, j = connector.get_position()
            new_i = i + I_DIR[num_dir]
            new_j = j + J_DIR[num_dir]
            #Si el otro participante del swap no está en Stand cancelar el swap
            if(other_connector.state != STATES.Stand):
                self.notify_not_swap(connector)
                continue
            #Si está iniciando el swap o cambió de dirección o objetivo
            if(not connector.prev_dir or not connector.prev_connector_to_swap or connector.state != STATES.Swaping
            or direction_to_int(connector.prev_dir) != num_dir or (connector.prev_connector_to_swap and connector.prev_connector_to_swap != other_connector)):
                connector.timer = max(connector.unit.get_move_cost,other_connector.unit.get_move_cost)
                connector.prev_connector_to_swap = other_connector
                connector.state = STATES.Stand
                connector.prev_dir = num_dir
            connector.timer -= 1
            if(connector.timer == 1):
                self.map[connector.get_position()].set_unit()
                self.map[other_connector.get_position()].set_unit()
                self.map[connector.get_position()].set_unit(other_connector)
                self.map[other_connector.get_position()].set_unit(connector)
                original_pos = connector.get_position()
                connector.update_position(original_pos)
                other_connector.update_position(connector)
                log.append(connector.get_position())
                log.append(other_connector.get_position())
    
    def exec_attack(self,log):
        for connector, pos_i, pos_j in self.attack_notifications:
            other_connector = self.map[pos_i,pos_j]
            was_alive = other_connector.is_connected()
            if(other_connector):
                if(other_connector.state == STATES.Stand):
                    other_connector.update_health_points(other_connector.get_health_points - connector.unit.get_damage)
                else:
                    if(randint(1,other_connector.unit.get_get_move_cost + connector.unit.get_get_move_cost) <= other_connector.unit.get_get_move_cost):
                        other_connector.update_health_points(other_connector.get_health_points - connector.unit.get_damage)

            if(was_alive and not other_connector.is_connected or not was_alive and other_connector.is_connected):
                log.append(other_connector.get_position())


    def is_valid_move(self,connector: Connector,num_dir : int):
        i, j = connector.get_position()
        new_i = i + I_DIR[num_dir]
        new_j = j + J_DIR[num_dir]
        height, width = self.map.shape
        if not validMove(new_i, new_j, height, width): return False
        cell_to_move = self.map[new_i,new_j]
        return cell_to_move.crossable(connector.unit) and not cell_to_move.get_unit
        
    
    def notify_move(self, connector : Connector, direction):
        num_dir = direction_to_int(direction)
        if self.is_valid_move(connector,num_dir):
            self.move_notifications.append((connector,num_dir))

    def notify_not_move(self,connector : Connector):
        connector.state = STATES.Stand
        connector.prev_dir = None
        connector.timer = 0

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

    def notify_not_swap(self,connector:Connector):
        connector.timer = 0
        connector.state = STATES.Stand
        connector.prev_connector_to_swap = None
        connector.prev_dir = None
        
    def notify_attack(self,connector:Connector,pos_i,pos_j):
        i, j = connector.get_pos()
        height, width = self.map.shape()
        if(pos_i < height and pos_j < width and discrete_distance(i,j,pos_i,pos_j) <= connector.get_attack_range):
            self.attack_notifications.append((connector,pos_i,pos_j))
    

    

     
    
    