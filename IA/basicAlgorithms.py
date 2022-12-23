import entities._map_elements as ME
import entities.utils as util
import numpy as np
from math import inf

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
             
        
                
                        
                        
                     
                    
                
                
                
            
    
    
    
    
    
        
    