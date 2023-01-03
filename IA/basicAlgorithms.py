from itertools import chain
from queue import PriorityQueue
import random
from ._definitions import *
import entities.utils as util
import numpy as np
from math import inf

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

def whcastar_search(goals: list[(tuple, tuple)], rrastar_list: list[RRAstar], roadmap, w: int):
    reservation_table = {}

    for i in range(len(goals)):
        connector, target = goals[i]
        start = connector.get_position()
        rrastar_list[i] = init_rrastar(start=start, target=target, roadmap=roadmap) if rrastar_list[i] is None else rrastar_list[i]
        reservation_table[(start, 0)] = connector
    paths = {}
    for i in range(len(goals)):
        connector, target = goals[i]
        start = connector.get_position()
        rrastar_instance = rrastar_list[i]
        path = start_whcastar_search(reservation_table=reservation_table, roadmap=roadmap, rrastar_instance=rrastar_instance, start=start, target=target, w=w)
        paths[connector] = path
        for cell in path:
            reservation_table[cell] = connector
    return paths

def slice(start_poss, width, height, filter, is_empty):
    def get_connectivities(slice, filter, is_empty):
        cell_parts = []
        start = -1
        for i, point in enumerate(slice):
            if not filter(*point): continue
            if not is_empty(*point) and start != -1:
                cell_parts.append((start, i))
                start = -1
            elif is_empty(*point) and start == -1:
                start = i
        if start != -1:
            cell_parts.append((start, len(slice)))
        return cell_parts

    def get_adjacency_matrix(left_parts, right_parts):
        return np.array([[min(left_part[1], right_part[1]) - max(left_part[0], right_part[0]) > 0 
                    for right_part in right_parts] 
                        for left_part in left_parts])

    decomposed = np.zeros(shape=(width, height), dtype = int) # black

    last_cells = []
    last_cell_parts = []
    last_cell_len = total_cells = 0
    cells_area = []
    cells_adjacency = {}

    dx, dy = start_poss
    for y in range(height):
        slice = [(dx + x, dy + y) for x in range(width)]
        cell_parts = get_connectivities(slice, filter, is_empty)
        cell_len = len(cell_parts)
        current_cells = []

        if last_cell_len == 0:    
            for _ in cell_parts:
                total_cells += 1
                cells_area.append(0)
                current_cells.append(total_cells)
        elif cell_len != 0:
            matrix = get_adjacency_matrix(last_cell_parts, cell_parts)
            current_cells = [0] * len(cell_parts)

            for i in range(last_cell_len):
                connectivities = [j for j, condition in enumerate(matrix[i, :]) if condition]
                connec_len = len(connectivities)
                if connec_len == 1:
                    current_cells[connectivities[0]] = last_cells[i]
                elif connec_len > 1:
                    for j in connectivities:
                        i_adj, j_adj = cells_adjacency[i], cells_adjacency[j]
                        
                        if i_adj is None:
                            cells_adjacency[i] = []
                        if j_adj is None:
                            cells_adjacency[j] = []
                        
                        i_adj.append(j)
                        j_adj.append(i)

                        total_cells += 1
                        cells_area.append(0)
                        current_cells[j] = total_cells

            for j in range(cell_len):
                connec_len = np.sum(matrix[:, j])
                if connec_len > 1 or connec_len == 0:
                    total_cells += 1
                    cells_area.append(0)
                    current_cells[j] = total_cells

        for cell, slice in zip(current_cells, cell_parts):
            decomposed[slice[0]: slice[1], y] = cell
            cells_area[cell - 1] += slice[1] - slice[0]

        last_cell_len = cell_len
        last_cell_parts = cell_parts
        last_cells = current_cells

    adjacency_matrix = np.ndarray(shape=(len(cells_area), len(cells_area)), dtype=bool)

    for i in range(len(cells_area)):
        for j in cells_adjacency[i]:
            adjacency_matrix[i, j] = True
    
    return cells_area, decomposed, adjacency_matrix

def Boustrophedon_path(cell, color, descomposition):
    start, floor, ceiling = cell
    x, y = start
    vertical_direction = "UP" if descomposition[x - 1, y] == color else "DOWN"
    horizontal_direction = -1 if descomposition[x, y - 1] == color else 1

    path = []
    index = 0 if horizontal_direction > 0 else len(floor) - 1
    end = 0 if horizontal_direction < 0 else len(floor) - 1
    while horizontal_direction * index < horizontal_direction * end:
        min_y = ceiling[index]
        max_y = floor[index]
        if vertical_direction == "UP" :
            path.append((index + x, min_y))
            path.append((index + x, max_y))
            y = ceiling[index + horizontal_direction]
            vertical_direction = "DOWN"
        else :
            path.append((index + x, max_y))
            path.append((index + x, min_y))
            y = floor[index + horizontal_direction]
            vertical_direction = "UP"
        path.append((index + x, y))
        path.append((index + x + horizontal_direction, y))
        index += 1

    path_length = 0
    prev_x, prev_y = x, y
    for x, y in path:
        path_length += x - prev_x + y - prev_y
        prev_x, prev_y = x, y

    start_point = path[0]
    end_point = path[-1]
    return start_point, end_point, path, path_length

def all_Boustrophedon_path(cells, descomposition):
    all_path = []
    dirs = [(I_DIR[dirvar.value - 1], J_DIR[dirvar.value - 1]) for dirvar in DIRECTIONS]
    for cell, color in cells:
        all_path.append(Boustrophedon_path(cell, color, descomposition))
    
    start = all_path[0][1]
    result = all_path[0][3].copy()
    result_length = all_path[0][4]
    for i in range(1, len(all_path)):
        end, next, other_path, other_length = all_path[i]
        firter = lambda x: x.state == end
        adj = lambda x: [NodeTree(state=(x.state[0] + dir_x, x.state[1] + dir_y), parent=x, path_cost=x.path_cost + 1) 
                            for dir_x, dir_y in dirs 
                                if util.validMove(x.state[0] + dir_x, x.state[1] + dir_y, *descomposition.shape) and 
                                    descomposition[x.state[0] + dir_x, x.state[1] + dir_y] != 0]
        he = lambda x: norma2(x.state, end)
        new_path = path_states(astar_search(NodeTree(state=start), firter, adj, he))
        result_length += len(new_path) + other_length
        result += new_path + other_path
        start = next
    return result, result_length

def end_path(roadmap, x, y):
    if roadmap.distance[x + 1, y] == 0 or roadmap.distance[x + 1, y + 1] == 0 or roadmap.distance[x + 1, y - 1] == 0:
        return False
    return True

def update_slice(slice_map, cell_up, cell_down, num_area, parent_up = None, parent_down = None):
    path_up = NodeTree(state=(cell_up, num_area), parent=parent_up)
    path_down = NodeTree(state=(cell_down, num_area), parent=parent_down)
    print_slice(cell_up, cell_down, slice_map, num_area)
    return path_up, path_down

def print_slice(a, b, slice_map, slice):
    for i in range(a[0], b[0]):
        slice_map[i, a[1]] = slice

def jump_obstacles(roadmap, x, y, height, color):
    i = x
    while i < height and (roadmap.distance[i, y] == 0 or roadmap.color[i, y] not in color):
        i += 1
    return i - 1

#*Clase que maneja todo lo relacionado con el roadmap basado en el diagrama Voronoi
class RoadMap:
    
    def __init__(self, worldMap, unit) -> None:
        self.unit = unit
        self.vertex, self.distance, self.color, self.roads, self.areas, self.adjacents, self.size = RoadMap.RoadMapGVD(worldMap,unit)
        
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
        return vertex, distance, color, roads, vertex_area, adj, size


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

