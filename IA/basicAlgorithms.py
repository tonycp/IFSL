from queue import PriorityQueue
from ._definitions import *
import entities.utils as util
import numpy as np
from math import inf


def breadth_first_search(node: Node, filter, reached: set[Node] = None, h: int = -1, expand=default_expand) -> list[Node]:
    frontier = [(node, 1)]
    reached = reached or set([node])
    result = []
    while frontier:
        node, nh = frontier.pop(0)
        if nh < h:
            return result
        if filter(node.state):
            result.append(node)
        for child in expand(node):
            if child not in reached:
                reached.add(child)
                frontier.append(child)
    return result


def best_first_search(node: NodeTree, filter, f=default_cost, expand=default_expand) -> Node:
    "Search nodes with minimum f(node) value first."
    frontier = PriorityQueue([node], key=f)
    reached = {node.state: node}
    while frontier:
        node = frontier.get()
        if filter(node.state):
            return node
        for child in expand(node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached.add(child)
                frontier.put(child)
    return None


def best_first_tree_search(node: NodeTree, filter, f=default_cost, expand=default_expand) -> Node:
    "A version of best_first_search without the `reached` table."
    frontier = PriorityQueue([node], key=f)
    while frontier:
        node = frontier.pop()
        if filter(node.state):
            return node
        for child in expand(node):
            if not is_cycle(child):
                frontier.add(child)
    return failure


def default_heuristic_cost(_): return 0
def astar_cost(he): return lambda n: default_cost(n) + he(n)


def astar_search(node: NodeTree, is_goal, expand, he=default_heuristic_cost) -> Node:
    """Search nodes with minimum f(n) = g(n) + he(n)."""
    return best_first_search(node, is_goal, f=astar_cost(he), expand=expand)


def astar_tree_search(node: NodeTree, is_goal, expand, he=default_heuristic_cost) -> Node:
    """Search nodes with minimum f(n) = g(n) + he(n), with no `reached` table."""
    return best_first_tree_search(node, is_goal, f=astar_cost(he), expand=expand)

class RoadMap:
    
    def __init__(self, worldMap, unit) -> None:
        self.unit = unit
        self.vertex, self.distance, self.color = self.RoadMapGVD(worldMap,unit)
    
    def RoadMapGVD(worldMap, unit):
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

        distance, queue = initialize()

        color = DFS(queue, distance)
        queue = add_border(worldMap.shape) + queue
        edges = set()
        roads = {}
        vertex = {}
        adj = {}

        while(len(queue) > 0):

            x, y, d = queue.pop(0)

            # si ya la casilla pertenece a una arista o vertice no tiene sentido expandirla
            if(util.validMove(x, y, worldMap.shape[0], worldMap.shape[1]) and color[x, y] is not None and len(color[x, y]) > 1):
                continue

            for i in range(len(util.I_DIR)): # Por cada direccion de expansion hacer
                dx = x + util.I_DIR[i]
                dy = y + util.J_DIR[i]

                # Comprobar si es posible expandirse en esa direccion
                if util.validMove(dx, dy, worldMap.shape[0], worldMap.shape[1]):
                    currentcolor = [-1] if x==-1 else [-2] if y==-1 else [-3] if x== worldMap.shape[0] else [-4] 
                    is_border = True
                    if util.validMove(x, y, worldMap.shape[0], worldMap.shape[1]):
                        currentcolor = color[x, y]
                        is_border = False
                    # Si la casilla aun no pertenece a ningun plano se marca como perteneciente al mismo plano que su adyacente
                    if distance[dx, dy] == inf:
                        distance[dx, dy] = d + 1
                        color[dx, dy] = currentcolor.copy()
                        queue.append((dx, dy, d + 1))
                        continue

                    # Si la casilla ya pertenecia a algun plano
                    elif (len(color[dx, dy]) == 1 and color[dx, dy][0] == currentcolor[0]) or is_border:
                        continue  # Y este plano es el mismo que el de su adyacente es que recien fue visitada
                    

                    if  currentcolor[0] not in color[dx, dy]:
                        # Si el vertice no separaba a este plano se agrega que lo separa
                        
                        # for col in color[dx,dy]:
                        #     r = road[(math.min(col,currentcolor[0]))]
                            
                        color[dx, dy] += currentcolor

                        # Si con el agrego de area la casilla arista ahora pasa a ser nodo vertice del GVD
                        if len(color[dx, dy]) >= 3:
                            node = vertex.get((dx, dy)) or Node(
                                (dx, dy), adjlist=[])  # obtengo en vertice

                            for j in range(len(util.I_DIR)):
                                dxx = dx + util.I_DIR[j]
                                dyy = dy + util.J_DIR[j]

                                if (dxx, dyy) in vertex.keys():
                                    if node not in vertex[(dxx, dyy)].adjlist:
                                        vertex[(dxx, dyy)].adjlist.append(node)
                                        node.adjlist.append(vertex[(dxx, dyy)])
                                elif (dxx, dyy) in edges:
                                    min = color[dxx, dyy][0]
                                    max = color[dxx, dyy][1]
                                    if min > max:
                                        (min, max) = (max, min)
                                    adj1 = adj.get((min, max)) or []

                                    if node not in adj1:
                                        for v in adj1:
                                            if node in v.adjlist: continue
                                            v.adjlist.append(node)
                                            node.adjlist.append(v)

                                        adj1.append(node)
                                        adj[(min, max)] = adj1

                            vertex[(dx, dy)] = node
                    edges.add((dx, dy))
        return vertex, distance, color


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
