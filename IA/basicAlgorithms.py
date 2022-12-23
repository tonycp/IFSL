
from queue import PriorityQueue
from ._definitions import *
import entities.utils as util
import numpy as np
from math import inf

DIR = [(0,1),(1,0),(0,-1),(-1,0)]

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
    edges = set()
    vertex ={}
    adj = {}
    
    
    while(queue.count > 0):

        x,y,d = queue.pop
        
        if(len(color[x,y]) > 1 ): #si ya la casilla pertenece a una arista o vertice no tiene sentido expandirla
            continue
        
        for i in DIR: #Por cada direccion de expansion hacer
            
            dx,dy = DIR[i]
            dx =x + dx
            dy =y + dy            
            
            if util.validMove(dx,dy,worldMap.shape(0), worldMap.shape(0)): #Comprobar si es posible expandirse en esa direccion
                
                if distance[dx,dy] == inf: #Si la casilla aun no pertenece a ningun plano se marca como perteneciente al mismo plano que su adyacente 
                    distance[dx,dy] = d + 1
                    color[dx,dy] = color[x,y] 
                    queue.append(dx,dy, d+1)
                    continue
                
                elif len(color[dx,dy]) == 1 and  color[dx,dy][0] == color[x,y][0]: #Si la casilla ya pertenecia a algun plano
                    continue                      #Y este plano es el mismo que el de su adyacente es que recien fue visitada
                
                if not color[x,y][0] in color[dx,dy]: 
                    color[dx,dy]+= color[x,y] #Si el vertice no separaba a este plano se agrega que lo separa
                    if len(color[dx,dy]) >= 3: #Si con el agrego de area la casilla arista ahora pasa a ser nodo vertice del GVD 
                        node =  vertex.get((dx,dy)) or Node((dx,dy),adjlist=[]) #obtengo en vertice 
                        
                        for j in util.X_DIR:
                            dxx =dx + util.X_DIR[j]
                            dyy =dy +  util.Y_DIR[j]
                            
                            if (dxx,dyy) in vertex.keys():
                                vertex[(dxx,dyy)].adjlist.append(node)
                                node.adjlist.append(vertex[(dxx,dyy)])
                            elif (dxx,dyy) in edges:
                                min =  color[dxx,dyy][0]
                                max =  color[dxx,dyy][1]
                                if min < max: 
                                    (min,max) = (max,min)
                                adj1 = adj.get((min, max)) or []
                                
                                for v in adj1:
                                    v.adjlist.append(node)
                                    node.adjlist.append(v)
                                
                                adj1.append(node)
                                adj[(min,max)] = adj1
                                
                        vertex[(dx,dy)] = node
                edges.add((dx,dy))
    
        
                
        
    

                    

        
def DFS(obstacles, distance):
    lenx = distance.shape(0)
    leny = distance.shape(1)
    
    visited = np.empty(shape=distance.shape, dtype=[])
    color = 0
    
    for (i,j) in obstacles:
        if visited[i,j]== 0:
            color +=1
            internalDFS((i,j),visited,[color])
    
           
    def internalDFS(obstacle, visited, color):
        x,y = obstacle
        visited[dx,dy] = color
        for i in util.X_DIR:
    
            dx =x + util.X_DIR[i]
            dy =y +  util.Y_DIR[i]

            if util.validMove(dx,dy,lenx, leny) and visited[dx,dy]==0 and distance[dx,dy] == 0:
                internalDFS(obstacles[dx,dy],visited,[color])
    
    return visited        
             

