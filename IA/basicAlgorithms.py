from itertools import chain
from queue import PriorityQueue
from ._definitions import *
import entities.utils as util
import numpy as np
from math import inf

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

#*Clase que maneja todo lo relacionado con el roadmap basado en el diagrama Voronoi
class RoadMap:
    
    def __init__(self, worldMap, unit) -> None:
        self.unit = unit
        self.vertex, self.distance, self.color, self.roads, self.areas, self.adjacents = RoadMap.RoadMapGVD(worldMap,unit)
        
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
        color = DFS(queue, distance)
        queue = add_border(worldMap.shape) + queue


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
        return vertex, distance, color, roads, vertex_area, adj


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

    return visited
