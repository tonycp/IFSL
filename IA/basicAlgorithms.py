from itertools import chain
from queue import PriorityQueue
import random
from ._definitions import *
import entities.utils as util
import numpy as np
from math import floor, inf

class FakeReservation:
    def __init__(self, reservation, current_time, max_time) -> None:
        self.fake_reservation = {}
        self.reservation = reservation
        self.current_time = current_time
        self.max_time = max_time

    def get(self, index):
        x, y, t = index
        time = self.current_time + t * self.max_time

        for i in range(self.max_time):
            value = self.reservation.get((x, y, time + i))
            if value is not None:
                return value
        return self.fake_reservation.get(index)

    def __set_item__(self, index, value):
        self.fake_reservation[index] = value

def hill_climbing(goals: list[tuple], cost, iter_count = 30):
    best_sol = goals.copy()
    random.shuffle(best_sol)
    cost_sol = cost(best_sol)
    change = True
    count = 0

    while(count < iter_count or change):
        change = False
        for i in range(len(best_sol)):
            for j in range(1, len(best_sol)):
                new_sol = best_sol.copy()
                new_sol[i], new_sol[j] = best_sol[j], best_sol[i]
                new_cost = cost(new_sol)
                if cost_sol > new_cost:
                    best_sol = new_sol
                    cost_sol = new_cost
                    change = True
        count += 1
    
    return best_sol

def breadth_first_search_problem(node: NodeTree, problem: Problem):
    filter = lambda n: problem.is_goal(n.state)
    newexpand = lambda n: expand(problem=problem, node=n)
    return breadth_first_search(node=node, filter=filter, adj=newexpand)


def breadth_first_search(node: NodeTree, filter, adj, reached: set[NodeTree] = None, h: int = inf) -> list[NodeTree]:
    frontier = [(node, 1)]
    reached = reached or set([node])
    result = []
    while len(frontier):
        node, nh = frontier.pop(0)
        if nh > h:
            return result
        if filter(node):
            result.append(node)
        for child in adj(node):
            if child not in reached:
                reached.add(child)
                frontier.append((child, nh+1))
    return result

def best_first_search_problem(node: NodeTree, problem: Problem, f=default_cost):
    filter = lambda n: problem.is_goal(n.state)
    newexpand = lambda n: expand(problem=problem, node=n)
    return best_first_search(node=node, filter=filter, adj=newexpand, f=f)

def best_first_search(node: NodeTree, filter, adj, f=default_cost) -> NodeTree:
    "Search nodes with minimum f(node) value first."
    frontier = PriorityQueue()
    frontier.put((f(node), node))
    reached = {node.state: node}
    while len(frontier.queue):
        _, node = frontier.get()
        if filter(node):
            return node
        for child in adj(node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.put((f(child), child))
    return None


def best_first_tree_search_problem(node: NodeTree, problem: Problem, f=default_cost):
    filter = lambda n: problem.is_goal(n.state)
    newexpand = lambda n: expand(problem=problem, node=n)
    return best_first_tree_search(node=node, filter=filter, adj=newexpand, f=f)

def best_first_tree_search(node: NodeTree, filter, adj, f=default_cost) -> NodeTree:
    "A version of best_first_search without the `reached` table."
    frontier = PriorityQueue()
    frontier.put((f(node), node))
    while len(frontier):
        _, node = frontier.get()
        if filter(node):
            return node
        for child in adj(node):
            if not is_cycle(child):
                frontier.put((f(child), child))
    return failure


def default_heuristic_cost(_): return 0
def astar_cost(g,he): return lambda n: g(n) + he(n)

def astar_search_problem(node: NodeTree, problem: Problem, he=default_heuristic_cost, g=default_cost):
    is_goal = lambda n: problem.is_goal(n.state)
    newexpand = lambda n: expand(problem=problem, node=n)
    return astar_search(node=node, is_goal=is_goal, adj=newexpand, he=he, g=g)

def astar_search(node: NodeTree, is_goal, adj, he=default_heuristic_cost, g=default_cost) -> NodeTree:
    """Search nodes with minimum f(n) = g(n) + he(n)."""
    return best_first_search(node, is_goal, adj=adj, f=astar_cost(g, he))


def astar_tree_search_problem(node: NodeTree, problem: Problem, he=default_heuristic_cost, g=default_cost):
    is_goal = lambda n: problem.is_goal(n.state)
    newexpand = lambda n: expand(problem=problem, node=n)
    return astar_tree_search(node=node, is_goal=is_goal, adj=newexpand, he=he, g=g)

def astar_tree_search(node: NodeTree, is_goal, adj, he=default_heuristic_cost, g=default_cost) -> NodeTree:
    """Search nodes with minimum f(n) = g(n) + he(n), with no `reached` table."""
    return best_first_tree_search(node, is_goal, adj=adj, f=astar_cost(g, he))

class RRAstar:
    def __init__(self, start, rrastar_problem, he=default_heuristic_cost, g=default_cost) -> None:
        self.f = astar_cost(g=g, he=he)
        self.frontier = PriorityQueue()
        self.frontier.put((self.f(start), start))
        self.reached = {start.state: start}
        self.filter = lambda n, p: n.state == p.state[0]
        self.adj = lambda n: expand(problem=rrastar_problem, node=n)

    def resume_rrastar(self, p):
        while len(self.frontier.queue):
            _, node = self.frontier.get()
            if self.filter(node, p):
                return node
            for child in self.adj(node):
                s = child.state
                if s not in self.reached or child.path_cost < self.reached[s].path_cost:
                    self.reached[s] = child
                    self.frontier.put((self.f(child), child))
        return None

def whcastar_heuristic(rrastar_instance: RRAstar, node: NodeTree):
    if rrastar_instance.reached.get(node.state[0]) is not None:
        return len(rrastar_instance.reached[node.state[0]])
    cost = len(rrastar_instance.resume_rrastar(node))
    if cost != 0:
        return cost
    return inf

def init_rrastar(start, target, roadmap):
    rrastar_problem = FindVertex(roadmap=roadmap, goals=[start])
    he = lambda n: util.norma2(start, n.state)
    rrastar_instance = RRAstar(start=NodeTree(state=target), rrastar_problem=rrastar_problem, he=he)
    return rrastar_instance

def start_whcastar_search(reservation_table: dict, roadmap, rrastar_instance: RRAstar, start, target, w: int):
    agent_problem = FindWithAgent(reservation_table=reservation_table, roadmap=roadmap, goals=[(target, w)])
    start_node = NodeTree(state=(start, 0))
    he = lambda n: whcastar_heuristic(rrastar_instance=rrastar_instance, node=n)
    filter = lambda n: agent_problem.is_goal(n.state) or n.state[1] == w
    adj = lambda n: chain(expand(agent_problem, n), add_edge(n=n, reservation_table=reservation_table, rrastar_instance=rrastar_instance, w=w))
    node = astar_search(node=start_node, is_goal=filter, adj=adj, he=he)
    return path_states(node=node)

def add_edge(n, reservation_table: dict, rrastar_instance: RRAstar, w):
    node = rrastar_instance.reached.get(n.state[0])
    time = len(node)
    if time > w:
        return []
    current_node = n
    id_1 = reservation_table.get((n.state[0], n.state[1] + 1))
    for item in path_states(node)[-2::-1]:
        id_2 = reservation_table.get((item, current_node.state[1]))
        state = item, current_node.state[1] + 1
        if reservation_table.get(state) is not None or (id_1 is not None and id_1 == id_2):
            return []
        id_1 = reservation_table.get((item, state[1] + 1))
        current_node = NodeTree(state=state, parent=current_node, path_cost=current_node.path_cost + 1)
    return [current_node]

def whcastar_search(connectors, goals: list[(tuple, tuple)], rrastar_list: list[RRAstar], roadmap, w: int, reservation_table):
    reservation_table = {} if not reservation_table else reservation_table

    for i in range(len(goals)):
        start, target = goals[i]
        rrastar_list[i] = init_rrastar(start=start, target=target, roadmap=roadmap) if rrastar_list[i] is None else rrastar_list[i]
        reservation_table[(start, 0)] = connectors[i]
    paths = {}
    for i in range(len(goals)):
        start, target = goals[i]
        rrastar_instance = rrastar_list[i]
        path = start_whcastar_search(reservation_table=reservation_table, roadmap=roadmap, rrastar_instance=rrastar_instance, start=start, target=target, w=w)
        paths[connectors[i]] = path
        for cell in path:
            reservation_table[cell] = connectors[i]
    return paths

def slice(start_poss, width, height, filter, is_empty):
    def get_connectivities(slice, filter, is_empty):
        cell_parts = []
        valid = False
        start = -1
        for i, point in enumerate(slice):
            if not (is_empty(*point) and filter(*point)) and start != -1:
                cell_parts.append((start, i))
                start = -1
            elif is_empty(*point) and filter(*point) and start == -1:
                start = i
                valid = True
        if start != -1:
            cell_parts.append((start, len(slice)))
        return cell_parts, valid

    def get_adjacency_matrix(left_parts, right_parts):
        return np.array([[min(left_part[1], right_part[1]) - max(left_part[0], right_part[0]) > 0 
                    for right_part in right_parts] 
                        for left_part in left_parts])

    decomposed = np.zeros(shape=(width, height), dtype = int) # black

    last_cells = []
    last_cell_parts = []
    last_cell_len = total_cells = 0
    cell_size = {}
    cell_floor = {}
    cell_ceiling = {}

    dx, dy = start_poss
    for x in range(width):
        slice = [(dx + x, dy + y) for y in range(height)]
        cell_parts, valid = get_connectivities(slice, filter, is_empty)
        cell_len = len(cell_parts)
        current_cells = []

        if not valid:
            last_cell_len = 0
            last_cell_parts = []
            last_cells = []
            continue

        if last_cell_len == 0:
            for _ in cell_parts:
                total_cells += 1
                current_cells.append(total_cells)
        elif cell_len != 0:
            matrix = get_adjacency_matrix(last_cell_parts, cell_parts)
            current_cells = [0] * len(cell_parts)

            for j in range(cell_len):
                connec_len = np.sum(matrix[:, j])
                if connec_len > 1 or connec_len == 0:
                    if current_cells[j] != 0:
                        continue
                    total_cells += 1
                    current_cells[j] = total_cells

            for i in range(last_cell_len):
                connectivities = [j for j, condition in enumerate(matrix[i, :]) if condition]
                connec_len = len(connectivities)
                if connec_len == 1 and current_cells[connectivities[0]] == 0:
                    current_cells[connectivities[0]] = last_cells[i]
                elif connec_len > 1:
                    for j in connectivities:
                        if current_cells[j] != 0:
                            continue
                        total_cells += 1
                        current_cells[j] = total_cells

        for cell, slice in zip(current_cells, cell_parts):
            decomposed[x, slice[0]:slice[1]] = cell
            size = cell_size.get(cell)
            floor = cell_floor.get(cell)
            ceiling = cell_ceiling.get(cell)

            if size is None:
                size = cell_size[cell] = [(dx + x, dy + slice[0]), (dx + x, dy + slice[1] - 1), 
                                          (dx + x, dy + slice[0]), (dx + x, dy + slice[1] - 1)]
            if floor is None:
                floor = cell_floor[cell] = []
            if ceiling is None:
                ceiling = cell_ceiling[cell] = []

            size[3] = (dx + x, dy + slice[1] - 1)
            size[2] = (dx + x, dy + slice[0])
            floor.append(dy + slice[1] - 1)
            ceiling.append(dy + slice[0])

        last_cell_len = cell_len
        last_cell_parts = cell_parts
        last_cells = current_cells
            
        
    return cell_size, cell_floor, cell_ceiling, decomposed

def Boustrophedon_path(cell, color, descomposition):
    start, floor, ceiling = cell
    x, y = start
    vertical_direction = "UP" if descomposition[x - 1, y] == color else "DOWN"
    horizontal_direction = -1 if descomposition[x, y - 1] == color else 1

    path = []
    index = 0 if horizontal_direction > 0 else len(floor) - 1
    end = 0 if horizontal_direction < 0 else len(floor) - 1
    while horizontal_direction * index < horizontal_direction * end:
        prev = ceiling if vertical_direction == "UP" else floor
        next = floor if vertical_direction == "DOWN" else ceiling
        min_y = prev[index]
        max_y = next[index]
        
        path.append((index + x, min_y))
        for i in range(min_y, max_y):
            path.append((index + x, i))
        y = next[index + horizontal_direction]

        for i in range(max_y, y + 1):
            path.append((index + x, i))
        path.append((index + x + horizontal_direction, y))

        vertical_direction = "DOWN" if vertical_direction == "UP" else "UP"
        index += 1

    start_point = path[0]
    end_point = path[-1]
    return start_point, end_point, path

def all_Boustrophedon_path(cells, descomposition):
    all_path = []
    dirs = [(I_DIR[dirvar.value - 1], J_DIR[dirvar.value - 1]) for dirvar in DIRECTIONS]
    for cell, color in cells:
        all_path.append(Boustrophedon_path(cell, color, descomposition))
    
    start = all_path[0][0]
    result = all_path[0][2].copy()
    for i in range(1, len(all_path)):
        end, next, other_path = all_path[i]
        firter = lambda x: x.state == end
        adj = lambda x: [NodeTree(state=(x.state[0] + dir_x, x.state[1] + dir_y), parent=x, path_cost=x.path_cost + 1) 
                            for dir_x, dir_y in dirs 
                                if util.validMove(x.state[0] + dir_x, x.state[1] + dir_y, *descomposition.shape) and 
                                    descomposition[x.state[0] + dir_x, x.state[1] + dir_y] != 0]
        he = lambda x: norma2(x.state, end)
        new_path = path_states(astar_search(NodeTree(state=start), firter, adj, he))
        result += new_path + other_path
        start = next
    return result

#*Clase que maneja todo lo relacionado con el roadmap basado en el diagrama Voronoi
class RoadMap:
    
    def __init__(self, worldMap, unit) -> None:
        self.unit = unit
        self.vertex, self.distance, self.color, self.roads, self.areas, self.adjacents, self.size, self.color_num = RoadMap.RoadMapGVD(worldMap,unit)
        
    def get_road(self, v1, v2):
        (x1,y1) = v1
        (x2,y2) = v2 
        
        col1 = set(self.color[x1,y1]) 
        col2 = set(self.color[x2,y2])
         
        intercept = list(col1.intersection(col2))
        
        if len(intercept) == 2:
            mini = min(intercept[0], intercept[1])
            maxi = max(intercept[0], intercept[1]) 
            return (mini,maxi) , self.roads[(mini,maxi)]
        
        return None , None
    
    def get_vertex_area(self,x,y):
        colors = self.color[x,y] 
        if len(colors) == 1:
            return self.areas.get(colors[0])
        elif len(colors) == 2:
            mini = min(colors[0],colors[1])
            maxi = max(colors[0], colors[1])
            return [node.state for node in self.adjacents[(mini,maxi)]]
        else:
            return (x,y)
        
    def get_road_cost(self, c1:int, c2:int):
        r = self.roads.get((c1,c2)) or  self.roads.get((c2,c1))
        if r: 
            return len(r)
        return inf
    
    def get_road_cost(self, v1, v2):
        _, r = self.get_road(v1,v2)
        if r: 
            return len(r)
        return inf
        
        
    
    def RoadMapGVD(worldMap, unit):
        
        #?######################### Structures ############################
        edges = set()
        roads = {}
        vertex = {}
        adj = {}
        vertex_area = {}
        size = {}
        
        #?######################## Inner Methods ###########################
        
        def initialize():
            queue = []
            distance = np.matrix(np.zeros(shape=worldMap.shape), copy=False)
            for i in range(0, worldMap.shape[0]):
                for j in range(0, worldMap.shape[1]):
                    if not worldMap[i, j].crossable(unit):
                        distance[i, j] = 0
                        queue.append((i, j, 0))
                    else:
                        distance[i, j] = inf

            return distance, queue

        def init_size(colors):
            for i in range(-3, colors + 1):
                num = i if i > 0 else i - 1
                size[num] = ((-inf, inf), (-inf, inf))
        
        def update_size(currentcolor, dx, dy):
            for num_color in currentcolor:
                newsize = size[num_color]
                item1 = max(newsize[0][0], dx), min(newsize[0][1], dx)
                item2 = max(newsize[1][0], dy), min(newsize[1][1], dy)
                size[num_color] = (item1, item2)

        def add_border(shape):
            border = [(-1, i, 0) for i in range(-1, shape[1])]
            border+= [(i, shape[1],0) for i in range(-1, shape[0])]
            border+= [(shape[0], i, 0) for i in range(0, shape[1] + 1)]
            border+= [(i, -1, 0) for i in range(0, shape[0] + 1)]
            return border
        
        def add_to_road(x,y, current, first):
            if first:
                minim = min(color[x,y][1],color[x,y][0]) 
                maxim = max(color[x,y][1],color[x,y][0])
                r = roads.get((minim ,maxim)) or set()
                r.add((x,y))
                roads[(minim, maxim)] = r
                return
                
            for col in color[x,y]:
                if col == current: continue
                minim = min(col,current) 
                maxim = max(col,current)
                r = roads.get((minim,maxim)) or set()
                r.add((x,y))
                roads[(minim,maxim)] = r
                
        def add_to_vertex(dx,dy):
            node = vertex.get((dx, dy)) or Node((dx, dy), actions=[])  # obtengo en vertice
            for j in range(len(util.I_DIR)):
                dxx = dx + util.I_DIR[j]
                dyy = dy + util.J_DIR[j]
                
                if (dxx, dyy) in vertex.keys():
                    if node not in vertex[(dxx, dyy)].actions:
                        vertex[(dxx, dyy)].actions.append(node)
                        node.actions.append(vertex[(dxx, dyy)])
                    
                elif (dxx, dyy) in edges and len(set(color[dx,dy]).intersection(color[dxx,dyy]))>=2:
                    mini = min(color[dxx, dyy][0],color[dxx, dyy][1])
                    maxi = max(color[dxx, dyy][0],color[dxx, dyy][1])
                    adj1 = adj.get((mini, maxi)) or []
                    
                    if node not in adj1:
                        for v in adj1:
                            if node in v.actions: continue
                            v.actions.append(node)
                            node.actions.append(v)
                        
                        adj1.append(node)
                        adj[(mini, maxi)] = adj1
            vertex[(dx, dy)] = node
           
        def add_to_area(x,y,current,first):
            if first:
                for col in color[x,y]:
                    a = vertex_area.get(col) or set()
                    a.add((x,y))
                    vertex_area[col] = a
                return
            
            a = vertex_area.get(current) or set()
            a.add((x,y))
            vertex_area[current] = a
            
              
        
        #?##############################################################################             

        distance, queue = initialize()
        color, num_colors = DFS(queue, distance)
        queue = add_border(worldMap.shape) + queue
        init_size(num_colors)

        while(len(queue) > 0):

            x, y, d = queue.pop(0)

            # si ya la casilla pertenece a una arista o vertice no tiene sentido expandirla
            if(util.validMove(x, y, worldMap.shape[0], worldMap.shape[1]) and color[x, y] is not None and len(color[x, y]) > 1):
                if len(color[x, y]) >= 3:
                    add_to_vertex(x,y)
                continue                

            for i in range(len(util.I_DIR)): # Por cada direccion de expansion hacer
                dx = x + util.I_DIR[i]
                dy = y + util.J_DIR[i]

                # Comprobar si es posible expandirse en esa direccion
                if util.validMove(dx, dy, worldMap.shape[0], worldMap.shape[1]):
                    currentcolor = [-1] if x==-1 else [-2] if y==-1 else [-3] if x== worldMap.shape[0] else [-4] 
                    # is_border = True
                    
                    if util.validMove(x, y, worldMap.shape[0], worldMap.shape[1]):
                        currentcolor = color[x, y]
                        # is_border = False
                    # Si la casilla aun no pertenece a ningun plano se marca como perteneciente al mismo plano que su adyacente
                    
                    update_size(currentcolor, dx, dy)

                    if distance[dx, dy] == inf:
                        distance[dx, dy] = d + 1
                        color[dx, dy] = currentcolor.copy()
                        queue.append((dx, dy, d + 1))           

                        continue

                    # Si la casilla ya pertenecia a algun plano
                    elif (len(color[dx, dy]) == 1 and color[dx, dy][0] == currentcolor[0]):
                        continue  # Y este plano es el mismo que el de su adyacente es que recien fue visitada
                    

                    if  currentcolor[0] not in color[dx, dy]:
                        # Si el vertice no separaba a este plano se agrega que lo separa
                        color[dx, dy] += currentcolor
                        add_to_road(dx,dy,currentcolor[0], len(color[dx,dy])== 2) 

                        # Si con el agrego de area la casilla arista ahora pasa a ser nodo vertice del GVD
                        if len(color[dx, dy]) >= 3:
                               add_to_road(dx,dy,currentcolor[0],len(color[dx, dy]) == 3)
                               add_to_area(dx,dy,currentcolor[0],len(color[dx, dy]) == 3)

                    queue.append((dx, dy, d + 1))           
                    edges.add((dx, dy))
        return vertex, distance, color, roads, vertex_area, adj, size, num_colors + 4


def DFS(obstacles, distance):
    lenx = distance.shape[0]
    leny = distance.shape[1]

    visited = np.ndarray(shape=(lenx, leny), dtype=list)
    color = 0

    def internalDFS(obstacle, visited, color):
        x, y, _ = obstacle
        visited[x, y] = color.copy()
        for i in range(len(util.I_DIR)):

            dx = x + util.I_DIR[i]
            dy = y + util.J_DIR[i]

            if util.validMove(dx, dy, lenx, leny) and visited[dx, dy] is None and distance[dx, dy] == 0:
                internalDFS((dx, dy, _), visited, color)

    for (i, j, _) in obstacles:
        if visited[i, j] is None:
            color += 1
            internalDFS((i, j, _), visited, [color])

    return visited, color

def knuth_shuffle(arr):
    result = [0]*len(arr)
    for i in range(len(arr)):
        index = randint(0,len(arr) - 1 - i)
        result[i] = arr[index]
        temp = arr[index]
        arr[index] = arr[len(arr) - 1 - i]
        arr[len(arr) - 1 - i] = temp
    return result

def knuth_shuffle_many(length,count):
    l = [i for i in range(0,length)]
    return [knuth_shuffle(l) for _ in range(count)]

def create_knuth_shuffle_pob_generator(length,count):
    def knuth_shuffle_pob_generator():
        return knuth_shuffle_many(length,count)
    return knuth_shuffle_pob_generator

def get_min(l,fitness_func):
    best = None
    best_fitness = np.infty
    for item in l:
        current_fitness = fitness_func(item)
        if(current_fitness < best_fitness):
            best_fitness = current_fitness
            best = item
    return best, best_fitness

class RouletteWheel_ParentSelector:
    def __init__(self,gen):
        self.gen = gen
        self.sum = sum([item[1][0] for item in gen])
        self.p = [item[1][0]/self.sum for item in gen]
        self.arange = np.arange(0,len(self.gen))

    def get_parents_index(self):
        parent_A_index = np.random.choice(self.arange, p = self.p)
        parent_B_index = np.random.choice(self.arange, p = self.p)
        return parent_A_index, parent_B_index

def find_first_non_marked(left,mark,start_index,cells):
    inc = 1
    if(left):
        inc = -1
    current_index = start_index + inc
    while(0 <= current_index < len(mark)):
        if(mark[(cells[current_index])]):
            current_index += inc
        else:
            return current_index
    return None


def create_crossover_cost(distance):
    def best_cost(current_cell,current_rotation,next_cell):
        area_list = [next_cell[rot].area for rot in range(4)]
        best_area = min(area_list)
        best_cost = np.infty
        best_rot = -1
        current_end = current_cell[current_rotation].end if current_cell else None
        for rot in range(4):
            cost = area_list[rot] - best_area + distance(current_end,next_cell[rot].start)
            if(cost < best_cost):
                best_cost = cost
                best_rot = rot
        return best_cost,best_rot
    return best_cost


def create_crossover_operator(cost_func):
    def crossover_operator(parent_A : list,parent_B: list,cell_info: list):
        mark = [False]*len(parent_A)
        result_cell = []
        result_rotation = []
        #Selecting start_cell
        index_A = randint(0,len(parent_A) - 1)
        cell_index = parent_A[index_A]
        index_B = parent_B.index(cell_index)
            
        result_cell.append(cell_index)
        mark[cell_index] = True
        _, prev_rotation = cost_func(None,None ,cell_info[cell_index])
        result_rotation.append(prev_rotation)
        #Seleccionar los siguientes
        
        for _ in range(1,len(parent_A)):
            first_left_A = find_first_non_marked(True,mark,index_A,parent_A)
            first_right_A = find_first_non_marked(False,mark,index_A,parent_A)
            first_left_B = find_first_non_marked(True,mark,index_B,parent_B)
            first_right_B = find_first_non_marked(False,mark,index_B,parent_B)
            possible_next_list =  [parent_A[first_left_A] if first_left_A != None else None,
                                   parent_A[first_right_A] if first_right_A != None else None,
                                   parent_B[first_left_B] if first_left_B != None else None,
                                   parent_B[first_right_B] if first_right_B != None else None]
            best_next_cost = np.infty
            best_next = -1
            best_next_rotation = -1
            last_cost = None
            for current_cell_index in possible_next_list:
                if(current_cell_index != None):
                    best_option_cost,best_option_rotation = cost_func(cell_info[result_cell[-1]],result_rotation[-1],cell_info[current_cell_index])
                    last_cost = best_option_cost
                    if(best_option_cost < best_next_cost):
                        best_next_cost = best_option_cost
                        best_next = current_cell_index
                        best_next_rotation = best_option_rotation
            result_cell.append(best_next)
            result_rotation.append(best_next_rotation)
            mark[best_next] = True
            index_A = parent_A.index(best_next)
            index_B = parent_B.index(best_next)
        return result_cell, result_rotation
    return crossover_operator
            
def mutation_sequence_creator(mutator_list):
    def mutator(child,cell_info):
        current_child = child
        for mut in mutator_list:
            current_child = mut(current_child,cell_info)
        return current_child
    return mutator

def swap_mutator(child,cell_info):
    cells, rotation = child
    index_A = randint(0,len(cells) -1)
    index_B = randint(0,len(cells) -1)
    swap_index(cells,index_A,index_B)
    swap_index(rotation,index_A,index_B)
    return (cells,rotation)


def swap_index(list,index_a,index_b):
    temp = list[index_a]
    list[index_a] = list[index_b]
    list[index_b] = temp
    

def invert_list(list,index_a,index_b):
    length = floor((index_b - index_a)/2)
    for i in range(length):
        swap_index(list,index_a + i, index_b - i)

def create_two_opt_mutator(distance, iter_count = 200):
    def two_opt_mutator(child,cell_info):
        my_iter_count = iter_count
        cells_index, rotation = child
        cells = [cell_info[index][rotation[index]] for index in cells_index]
        change = True
        while(change and my_iter_count > 0):
            change = False
            saliendo = False
            for i in range(len(cells)):
                prev_prev = cells[i - 1].end if i != 0 else None
                for j in range(i + 1,len(cells)):
                    post_post = cells[j + 1].start if j + 1 < len(child) else None
                    if(distance(prev_prev,cells[i].start) + distance(cells[j].end,post_post)
                     > distance(prev_prev,cells[j].start) + distance(cells[i].end,post_post)):
                        invert_list(cells,i,j)
                        invert_list(cells_index,i,j)
                        
                        change = True
                        saliendo = True
                        break
                if(saliendo):
                    break
            saliendo = False
            my_iter_count -= 1
        return cells_index,rotation
    return two_opt_mutator
        
def create_fitness_func(distance):
    def fitness_func(path: list[int], cell_info: list[int]):
        cell = cell_info[path[0]]
        mem_cost = np.ndarray(shape = (4,len(path)),dtype=int)
        mem_prev_rotation = np.ndarray(shape = (4,len(path)),dtype= int)
        for rot in range(4):
            mem_cost[rot,0] = distance(None,cell[rot].start) + cell[rot].area
        for i in range(1, len(path)):
            current_cell = cell_info[path[i]]
            prev_cell = cell_info[path[i - 1]]
            for current_rot in range(4):
                cell_cost = current_cell[current_rot].area
                best_cost = np.infty
                best_rot = -1
                for prev_rot in range(4):
                    cost = mem_cost[prev_rot,i-1] + distance(prev_cell[prev_rot].end,current_cell[current_rot].start) + cell_cost
                    if(cost < best_cost):
                        best_cost = cost
                        best_rot = prev_rot
                mem_cost[current_rot,i] = best_cost
                mem_prev_rotation[current_rot,i] = best_rot
        best_rotation = []
        best_rot = -1
        best_cost = np.infty
        for rot  in range(4):
            if(mem_cost[rot,len(path) - 1] < best_cost):
                best_cost = mem_cost[rot,len(path) - 1]
                best_rot = rot
        best_rotation.append(best_rot)
        for i in range(len(path) - 1,0,-1):
            best_rotation.insert(0,mem_prev_rotation[best_rotation[0],i])
        return best_cost, best_rotation
    return fitness_func




def GA(pob_generator, parents_selector, crossover_operator,mutation_operator, fitness_func,cell_info, top_generation = 100):
    current_gen = [(item,fitness_func(item,cell_info)) for item in pob_generator()]
    ((best,(best_fitness,best_rotation)),_) = get_min(current_gen,lambda item: item[1][0])
    # obtener el mejor
    for _ in range(top_generation):
        #crear la nueva generacion
        new_gen = []
        new_gen_parents_selector = parents_selector(current_gen)
        for _ in range(len(current_gen)):
            parent_A_index, parent_B_index = new_gen_parents_selector.get_parents_index()
            parent_A = current_gen[parent_A_index][0]
            parent_B = current_gen[parent_B_index][0]
            new_child, _ = mutation_operator(crossover_operator(parent_A,parent_B,cell_info),cell_info)
            new_child_fitness, new_child_rotation = fitness_func(new_child,cell_info)
            new_gen.append((new_child,(new_child_fitness,new_child_rotation)))
            if(new_child_fitness < best_fitness):
                best_fitness = new_child_fitness
                best_rotation = new_child_rotation
                best = new_child
        current_gen = new_gen
    return best,best_rotation

def create_min_distance(origin_pos):
    def min_distance(pos_A,pos_B):
        if(pos_A == None):
            pos_A = origin_pos
        if(pos_B == None):
            return 0
        i_A,j_A = pos_A
        i_B,j_B = pos_B
        return max(abs(i_A - i_B),abs(j_A - j_B))
    return min_distance

class Cell_Info:
    class Rotation_Info:
        def __init__(self, start, end, path, area):
            self.start = start
            self.end = end
            self.path = path
            self.area = area

    def __init__(self, cell_size, cell_ceiling, cell_floor):
        self.cell_size = cell_size
        self.cell_floor = cell_floor
        self.cell_ceiling = cell_ceiling
        self.rotations = []

        for rot in range(2):
            start, end, path_1, cell_area = self._get_path(cell_size[rot], rot)
            path_2 = path_1.copy()
            path_2.reverse()
            self.rotations.append(Cell_Info.Rotation_Info(start, end, path_1, cell_area))
            self.rotations.append(Cell_Info.Rotation_Info(end, start, path_2, cell_area))

    def _get_path(self, start, vertical_direction):
        path = []
        x, y = start
        index, end = 0, len(self.cell_floor)
        while index < end:
            prev, next = (self.cell_floor, self.cell_ceiling) if not vertical_direction else (self.cell_ceiling, self.cell_floor)

            path.append((index + x, prev[index]))
            if prev[index] != next[index]:
                path.append((index + x, next[index]))

            if index + 1 >= end:
                break
            elif not vertical_direction and next[index] > next[index + 1]:
                path.append((index + x, next[index + 1]))
            elif vertical_direction and next[index] < next[index + 1]:
                path.append((index + x, next[index + 1]))
            elif not vertical_direction and next[index] < next[index + 1]:
                path.append((index + x + 1, next[index]))
            elif vertical_direction and next[index] > next[index + 1]:
                path.append((index + x + 1, next[index]))

            index += 1
            y = next[index]
            vertical_direction = not vertical_direction

        path_length = 1
        prev_x, prev_y = path[0]
        for x, y in path:
            path_length += abs(x - prev_x + y - prev_y)
            prev_x, prev_y = x, y

        start_point = path[0]
        end_point = path[-1]
        return start_point, end_point, path, path_length

    def default_rotation_func(rot,cell_size):
        if(rot == 0):
            return cell_size[0],cell_size[3]
        elif (rot == 1):
            return cell_size[1],cell_size[2]
        elif (rot == 2):
            return cell_size[2],cell_size[1]
        elif (rot == 3):
            return cell_size[3],cell_size[0]


    def __getitem__(self,index):
        return self.rotations[index]