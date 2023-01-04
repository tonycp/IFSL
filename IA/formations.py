from ._definitions import Node, CSP_UncrossedAsignmentTime, evaluate_HillClimbingAsignment
from .basicAlgorithms import hill_climbing
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
                    updates.append((n,((nsx*abs(x) + xp) + abs(y)*ewx , (nsy*abs(x) + yp) + abs(y)*ewy)))
                    self.relations[n] = nsx*abs(x)  + abs(y)*ewx, nsy*abs(x)  + abs(y)*ewy 

        def get_directions(s1,s2,rot):
            ns = (0,0)
            ew = (0,0)
            if s1 >0:
                ns = direction_to_tuple(int_to_direction(abs(direction_to_int(DIRECTIONS.S) + rot)%8))
            elif s1<0:
                ns = direction_to_tuple(int_to_direction(abs(direction_to_int(DIRECTIONS.N) + rot)%8))
            if s2 > 0:
                ew = direction_to_tuple(int_to_direction(abs(direction_to_int(DIRECTIONS.E) + rot)%8))
            elif s2<0:
                ew = direction_to_tuple(int_to_direction(abs(direction_to_int(DIRECTIONS.W) + rot)%8))
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
        updates = []
        self.nodes[self.main].rotate(rot_times,self.nodes[self.main].position, updates)
        for key, pos in updates:
            self.nodes[key].rotate(rot_times,pos,updates)

    def asign_formation(self, start):
        goal = [pos.position for pos in self.nodes if pos.state != self.main]
        csp = CSP_UncrossedAsignmentTime(start, goal)
        csp.back_track_solving()
        if csp.asignment:
            from_to = [(y,x) for x,y in csp.asignment.items()]
        else:
            cost = lambda x: evaluate_HillClimbingAsignment(csp, start, x, alpha= 10)
            best = hill_climbing(goal, cost, 100)
            from_to = list(zip(start, best))
        return from_to

    @property
    def poss(self):
        return self.nodes[self.main].position

class TwoRowsFormation(Formation):
    
    def __init__(self, node, start):
        nodeN = node+1
        up_row_size = int((nodeN-1)/2) 
        midle_row = int(up_row_size/2) + 1
        edges =  {0: [(0, i) for i in range(1, nodeN)]}
        relative_positions = { 0: dict([(i, (0, i - midle_row)) for i in range(1, up_row_size +1)]+ [(i, (1, i - up_row_size - midle_row)) for i in range(up_row_size +1 ,nodeN)])}
        Formation.__init__(self,nodeN, edges, relative_positions, main_position=start)        
        
        
class OneFormation(Formation):
   def __init__(self, start):
        edges =  {0: [1]}
        relative_positions = { 0: {1: (0,0)}}
        Formation.__init__(self,2, edges, relative_positions, main_position=start)        
 