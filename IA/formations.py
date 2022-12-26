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
        
                
        def rotate(self, xa,ya, position, updates):
            self.position = position
            
            
            
            xp, yp = position
            if self.relations:
                for n in self.relations.keys(): 
                    x, y =self.relations[n]
                    npx, npy =  (xa*x + xp) + y*xa , (ya*x + yp) + y*ya 
                    updates[n] =(xa*x + xp) + y*xa , (ya*x + yp) + y*ya 
                    self.relations[n] = (xa*x, ya*y)

        
    def __init__(self, nodeN: int, edges:dict, relative_positions:dict, main:int = 0, main_position = None):
        self.nodes = []
        self.main = main 
        self.N = nodeN
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
        xa = I_DIR[angle]
        ya = J_DIR[angle]
        updates = {}
        self.nodes[self.main].rotate(xa,ya,self.nodes[self.main].position, updates)
        for key in updates.keys():
            pos = updates.pop(key)
            self.nodes[key].rotate(xa,ya,pos,updates)
            
            
        
        