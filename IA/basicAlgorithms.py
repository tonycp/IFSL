
from queue import PriorityQueue
from ._definitions import *
import entities._map_elements as ME
import entities.utils as util
import numpy as np
from math import inf

def best_first_search(problem: Problem, f: function):
    "Search nodes with minimum f(node) value first."
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure

def best_first_tree_search(problem: Problem, f: function):
    "A version of best_first_search without the `reached` table."
    frontier = PriorityQueue([Node(problem.initial)], key=f)
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            if not is_cycle(child):
                frontier.add(child)
    return failure

def astar_search(problem, he=None):
    """Search nodes with minimum f(n) = g(n) + he(n)."""
    he = he or problem.h
    return best_first_search(problem, f=lambda n: g(n) + he(n))

def Astar_tree_search(problem, he=None):
    """Search nodes with minimum f(n) = g(n) + he(n), with no `reached` table."""
    he = he or problem.h
    return best_first_tree_search(problem, f=lambda n: g(n) + he(n))



def RoadMapGVD(worldMap, unit):
    
    
    def initialize():
        queue = []
        distance = np.matrix(np.zeros(shape=worldMap.shape), copy=False)
        for i in range(0, worldMap.shape(0)):
            for j in range(0, worldMap.shape(1)):
                if not worldMap[i,j].crossable(unit):
                    distance[i,j] = 0
                    queue.append((i,j,0))
                else:
                    distance[i,j] = inf
        
        return distance, queue
    
    distance, queue = initialize()
    color = DFS(queue,distance)
    road = []
    vertex = []
    while(queue.count > 0):

        x,y,d = queue.pop
        for i in util.X_DIR:
            
            dx =x + util.X_DIR[i]
            dy =y +  util.Y_DIR[i]
            
            if util.validMove(dx,dy,worldMap.shape(0), worldMap.shape(0)):
                if distance[dx,dy] == inf:
                    distance[dx,dy] = d + 1
                    color[dx,dy] = color[x,y]
                    queue.append(dx,dy, d+1)
                elif not color[dx,dy] == color[x,y]: 
                    if (dx,dy) in road and not (dx,dy) in vertex:
                        vertex.append((dx,dy))
                    else: 
                        road.append((dx,dy))
    

        
def DFS(obstacles, distance):
    
    lenx = distance.shape(0)
    leny = distance.shape(1)
    
    visited = np.matrix(np.zeros(shape=distance.shape), copy=False)
    color = 0
    
    for (i,j) in obstacles:
        if visited[i,j]== 0:
            color +=1
            internalDFS((i,j),visited,color)
    
           
    def internalDFS(obstacle, visited, color):
        x,y = obstacle
        visited[dx,dy] = color
        for i in util.X_DIR:
    
            dx =x + util.X_DIR[i]
            dy =y +  util.Y_DIR[i]

            if util.validMove(dx,dy,lenx, leny) and visited[dx,dy]==0 and distance[dx,dy] == 0:
                internalDFS(obstacles[dx,dy],visited,color)
    
    return visited        
             

