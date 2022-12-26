from ._definitions import Node
from entities.utils import *

class Formation:
    class FormationNode(Node): 
        def __init__(self, state, adjlist=None, relations: dict= None):
            self.relations = relations
            self.position = None
            Node.__init__(self, state, adjlist)
        
        def update_relation(self, add: dict, remove):
            for r in remove:
                self.relations.pop(r)
            
            for key in add.keys():
                for a in add[key]:    
                    self.relations[key] = a 
        
        def update_position(self, x,y, updates):
            self.position = (x,y)
            if self.relations:
                for n in self.relations.keys():
                    x1,y1 = self.relations[n]
                    updates.append((n, (x+x1,y+y1)))
        
                
        def rotate(self, rot_times, position, updates):
            self.position = position
            xp, yp = position
            if self.relations:
                for n in self.relations.keys(): 
                    x, y =self.relations[n]
                    (nsx,nsy),(ewx,ewy) = Formation.FormationNode.get_directions(x,y, rot_times)
                    updates[n] = (nsx*x + xp) + y*ewx , (nsy*x + yp) + y*ewy 
                    self.relations[n] = nsx*x  + y*ewx, nsy*x  + y*ewy 

        def get_directions(s1,s2,rot):
            ns = None
            ew = None
            if s1 >0:
                ns = direction_to_tuple(int_to_direction(abs(direction_to_int(DIRECTIONS.S) + rot))%7)
            else:
                ns = direction_to_tuple(int_to_direction(abs(direction_to_int(DIRECTIONS.N) + rot)%7))
            if s2 >0:
                ew = direction_to_tuple(int_to_direction(abs(direction_to_int(DIRECTIONS.E) + rot))%7)
            else:
                ex = direction_to_tuple(int_to_direction(abs(direction_to_int(DIRECTIONS.W) + rot)%7))
            return ns, ew
        
    def __init__(self, nodeN: int, edges:dict, relative_positions:dict, main:int = 0, main_position = None, direction = DIRECTIONS.N):
        self.nodes = []
        self.main = main 
        self.N = nodeN
        self.dir = direction
        for n in range(0, nodeN):
            self.nodes.append(Formation.FormationNode(n,edges.get(n),relative_positions.get(n)))
        
        if main_position:
            self.set_in(*main_position)
    
    def set_in(self, x,y):
        updates = []
        self.nodes[self.main].update_position(x,y, updates)
        for key,(x,y) in updates:
            self.nodes[key].update_position(x,y,updates)
    
    def move(self, dx,dy):
        x,y = self.nodes[self.main].position 
        self.set_in(x+dx, y+dy)
        
    def rotate(self, angle: int):
        rot_times = angle - direction_to_int(self.dir) 
        self.dir = int_to_direction(angle)
        updates = {}
        self.nodes[self.main].rotate(rot_times,self.nodes[self.main].position, updates)
        for key in updates.keys():
            pos = updates.pop(key)
            self.nodes[key].rotate(rot_times,pos,updates)
            
            
        
        