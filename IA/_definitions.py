import numpy as np
import math
from entities.utils import DIRECTIONS, I_DIR, J_DIR, validMove, norma2

class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default actions cost is 1 for all states.
    When you create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, goals=None, **kwds): 
        self.goals=goals
        self.__dict__.update(**kwds)
        
    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state in self.goals
    def action_cost(self, s, a, s1): return 1
    def h(self, node):               return 0
    
    def __str__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.goals)

class ViewProblem(Problem):
    def __init__(self, roadmap, *args, **kwargs):
        self.roadmap = roadmap
        self.dim_x, self.dim_y = roadmap.distance.shape
        Problem.__init__(self, *args, **kwargs)

    def actions(self, state):
        """The actions executable in this state."""
        x, y = state
        dirs = [(dirvar.name, (I_DIR[dirvar.value - 1], J_DIR[dirvar.value - 1])) for dirvar in DIRECTIONS]
        return [(name, move) for name, move in dirs 
                                    if validMove(x + move[0], y + move[1], self.dim_x, self.dim_y)]

    def result(self, state, action):
        """The state that results from executing this action in this state."""
        x, y = state
        _, (x_dir, y_dir) = action
        return x + x_dir, y + y_dir

    def is_goal(self, state):
        """True if the goal level is in any one of the jugs."""
        return state in self.goals    

class FindVertex(ViewProblem):
    def actions(self, state):
        """The actions executable in this state."""
        x, y = state
        dirs = [(dirvar.name, (I_DIR[dirvar.value - 1], J_DIR[dirvar.value - 1])) for dirvar in DIRECTIONS]
        return [(name, move) for name, move in dirs 
                                    if validMove(x + move[0], y + move[1], self.dim_x, self.dim_y) and self.roadmap.distance[x + move[0], y + move[1]] != 0]

class FindWithAgent(FindVertex):
    def __init__(self, reservation_table: dict, *args, **kwargs):
        self.reservation_table = reservation_table
        FindVertex.__init__(self, *args, **kwargs)

    def actions(self, state):
        poss, time = state
        actions = FindVertex.actions(self, poss)
        actions.append(("wait", (0, 0)))
        
        newactions = []
        id_1 = self.reservation_table.get((poss, time + 1))
        for name, move in actions:
            s1 = self.result(state, (name, move))
            id_2 = self.reservation_table.get((s1[0], time))
            if self.reservation_table.get(s1) is None and (id_1 is None or id_1 != id_2):
                newactions.append((name, move))
        return newactions
    
    def result(self, state, action):
        return (FindVertex.result(self, state[0], action), state[1] + 1)
    
    def action_cost(self, s, a, s1):
        if a[0] == "wait" and s1 in self.goals and s1[1] < self.goals[0][1]:
            return 0
        if s1[1] != self.goals[0][1]:
            return 1
        return FindVertex.action_cost(self, s, a, s1)    
    
    def is_goal(self, state):
        return FindVertex.is_goal(self, state)
    

class FindVoronoiVertex(FindVertex):
    def __init__(self, roadmap, *args, **kwargs):
        self.roadmap = roadmap
        Problem.__init__(self, *args, **kwargs)

    def actions(self, state):
        """The actions executable in this state."""
        x, y = state
        return [(name, move) for name, move in FindVertex.actions(self, state) 
                            if len(set(self.roadmap.color[x + move[0], y + move[1]]).intersection(self.roadmap.color[x, y])) > 0]

class SliceProblem(FindVertex):
    def __init__(self, colors, *args, **kwargs):
        self.colors = colors
        FindVertex.__init__(self, *args, **kwargs)

    def actions(self, state):
        x, y = state
        return [(name, move) for name, move in FindVertex.actions(self, state) 
                            if len(set(self.roadmap.color[x + move[0], y + move[1]]).intersection(self.colors)) > 0]

class MoveVoronoiProblem(Problem):
    def __init__(self, start_costs, end_cost, start_end, roadmap, *args, **kwargs):
        self.start_costs = start_costs
        self.end_cost = end_cost
        self.start_end = start_end
        self.roadmap = roadmap
        Problem.__init__(self, *args, **kwargs)
    
    def actions(self, state):
        """The actions executable in this state."""
        if len(self.roadmap.color[state]) < 3:
            return self.roadmap.get_vertex_area(*state)
        # areas = self.roadmap.color[state]
        # for i in range(0, len(areas)):
        #     for j in range(i + 1, len(areas)):
        #         min = areas[i]
        #         max = areas[j]
        #         if min > max:
        #             (min, max) = (max, min)
        #         actions += self.roadmap.adjacents.get((min, max)) or []
        actions = [node.state for node in self.roadmap.vertex[state].actions]
        if state in self.start_end:
            actions.append(self.end_cost[state][-1])
        return actions

    def result(self, _, action):
        """The state that results from executing this action in this state."""
        return action

    def is_goal(self, state):
        """True if the goal level is in any one of the jugs."""
        return state in self.goals

    def action_cost(self, s, _, s1):
        if len(self.roadmap.color[s]) < 3:
            return len(self.start_costs[s1])
        elif len(self.roadmap.color[s1]) < 3:
            return len(self.end_cost[s])
        return self.roadmap.get_road_cost(s, s1)





class Node(object):
    "A Node in a search tree."
    def __init__(self, state, actions=None):
        self.state=state
        self.actions=actions

    def __repr__(self): return '<{}>'.format(self.state)
    def __eq__(self, __o: object) -> bool: return __o is not None and self.state == __o.state
    def __hash__(self) -> int: return hash(self.state)

class NodeTree(Node):
    def __init__(self, parent=None, path_cost=0, *args, **kwargs):
        self.parent=parent
        self.path_cost=path_cost
        Node.__init__(self, *args, **kwargs)
    
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost

failure = NodeTree(state='failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = NodeTree(state='cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.
def default_cost(n: NodeTree): return n.path_cost


def expand(problem: Problem, node: NodeTree):
    "Expand a node, generating the children nodes."
    s = node.state
    for actions in problem.actions(s):
        s1 = problem.result(s, actions)
        cost = node.path_cost + problem.action_cost(s, actions, s1)
        yield NodeTree(state=s1, parent=node, actions=actions, path_cost=cost)


def path_actions(node: NodeTree):
    "The sequence of actions to get to this node." 
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.actions]


def path_states(node: NodeTree):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state]


def is_cycle(node: NodeTree, k: int = 30):
    "Does this node form a cycle of length k or less?"
    def find_cycle(ancestor, k):
        return (ancestor is not None and k > 0 and
                (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1)))
    return find_cycle(node.parent, k)





#?###############################################################################################
#?                                          CSP                                                 #
#?###############################################################################################
class CSP:
    def __init__(self,domain, restrictions, variables, **kwds):
        self.asignment = {}
        self.domain = domain
        self.variables = variables
        self.restrictions = restrictions
    
    def back_track_solving(self, asignment =None):
        if asignment is None:
            asignment = {}
        if self.is_complete(asignment):
            self.asignment = asignment 
            return True
        var = self.get_unasigned_variable(asignment)
        
        domain_copy = dict([(i,j.copy()) for i,j in self.domain.items()])
        copy_asignment = dict([(i,j) for i,j in asignment.items()])
        for value in domain_copy[var]:
            if not self.is_valid_current_asign(value, var, copy_asignment): continue
            asignment[var] = value
            self.domain[var] = [value]
            
            if self.arc_consistence(self.get_queue()) and self.back_track_solving(asignment):
                return True

            asignment = dict([(i,j.copy()) for i,j in copy_asignment.items()])
            self.domain = dict([(i,j) for i,j in domain_copy.items()])
        return False
    
    def arc_consistence(self, queue):
        while len(queue):
            x ,y = list(queue).pop(0)
            queue = queue.difference([(x,y)])
            if self.remove_inconsistent_values(x,y):
               queue= queue.union([(z,x) for z in self.neighbors(x)])
            if not all(len(x)>0 for _,x in self.domain.items()): break
        return all(len(x)>0 for _,x in self.domain.items()) 
                    

            
        
    def is_complete(self, asignment): raise NotImplementedError
    def get_unasigned_variable(self, asignement): raise NotImplementedError 
    def is_valid_asignment(self, asignment): raise NotImplementedError
    def is_valid_current_asign(self, domain_value, unasigned_variable,asignment):raise NotImplementedError
    def neighbors(self, x):return [v for v in self.variables if v!=x]
    def get_queue(self): return set([(v , i)for v in self.variables for i in self.variables if i != v ])
        

class CSP_UncrossedAsignment(CSP):
    
    def __init__(self, domain, variables, **kwds):
        restrictions = dict([(v , [i for i in variables if i != v]) for v in variables])
        real_domain = dict([(v , [i for i in domain]) for v in variables])
        if len(domain) != len(variables): raise ValueError
        CSP.__init__(self, real_domain, restrictions, variables, **kwds)
        
    def is_complete(self, asignment):
        return all([asignment.get(var) is not None for var in self.variables])
    
    def get_unasigned_variable(self, asignement):
        for variable in self.variables:
            if asignement.get(variable) is None:
                return variable 
        return None
    
    def segment_interception(self,x1,x2,x3,x4,y1,y2,y3,y4):
        return CSP_UncrossedAsignment.intersect(x1,x2,x3,x4,y1,y2,y3,y4) and CSP_UncrossedAsignment.intersect(x3,x4,x1,x2,y3,y4,y1,y2)

    def is_valid_current_asign(self, domain_value, unasigned_variable,asignment):
        for (x1,y1), conector in asignment.items():
            (x2,y2) = conector.get_position() 
            (x4,y4) = domain_value.get_position()
            (x3,y3) = unasigned_variable
            if self.segment_interception(x1,x2,x3,x4,y1,y2,y3,y4):
                return False
        return True
    
    def is_valid_asignment(self,asignment):
        x =  asignment.items()
        for i in range(0,len(x)):
            (x1,y1),conector = x[i] 
            (x2,y2) = conector.get_position() 
            for j in range(i,len(x)):
                (x3,y3),conector = x[i] 
                (x4,y4) = conector.get_position()
                if self.segment_interception(x1,x2,x3,x4,y1,y2,y3,y4):
                    return False
        
        return True
    
    def remove_inconsistent_values(self, x, y):
        removed = False
        rem = []
        for val in self.domain[x]:
            x2,y2 = val.get_position()
            round = False
            for val2 in self.domain[y]:
                if round: break
                x4,y4 = val2.get_position()
                round |= val != val2 and not self.segment_interception(x[0],x2,y[0],x4,x[1],y2,y[1],y4)
            if not round:
                rem.append(val)
                removed = True
        
        for val in rem:
            self.domain[x].remove(val)
            
        return removed


    def intersect(x1,x2,x3,x4,y1,y2,y3,y4):
        dx, dy = x2-x1, y2-y1
        dx1, dy1 = x3-x1, y3-y1
        dx2, dy2 = x4-x2, y4-y2 
        return (dx*dy1 - dy*dx1) * (dx*dy2 - dy*dx2) <= 0       
    

class CSP_UncrossedAsignmentTime(CSP_UncrossedAsignment):
    def __init__(self, domain, variables, **kwds):
        CSP_UncrossedAsignment.__init__(self, domain, variables, **kwds)
    
    def segment_interception(self, x1, x2, x3, x4, y1, y2, y3, y4):
        
        if not CSP_UncrossedAsignment.segment_interception(self,x1, x2, x3, x4, y1, y2, y3, y4): return False

        #si el origen de cada conector es igual a su destino, son dos puntos y no se cruzan
        if (x1,y1)==(x2,y2) and (x3,y3) == (x4,y4): return False
        elif (x1,y1)==(x2,y2) or (x3,y3) == (x4,y4): #Si al menos un conector esta n su destino basa con ver que no este en medio del camino del otro
            return not (self.colinear_valid(x1,x2,x3,x4) or self.colinear_valid(y1,y2,y3,y4))


        if x1==x2 and x3==x4: #Si son dos rectas veticales se busca condicion de cruce en una linea en el plano rotado
            return not self.colinear_valid(y1,y2,y3,y4)
        
        if x1 == x2: 
            m2 = (y3-y4)/(x3-x4)
            n2 = y3 - m2*x3 

            y = m2*x1 + n2 
            x = x1

        elif x3 == x4: 
            m1 = (y1-y2)/(x1-x2)
            n1 = y1 - m1*x1 

            y = m1*x3 + n1 
            x = x3
        
        else:

            m1 = (y1-y2)/(x1-x2)
            m2 = (y3-y4)/(x3-x4)

            if m1==m2:
                return not self.colinear_valid(x1, x2, x3, x4)
            
            n1 = y1 - m1*x1
            n2 = y3 - m2*x3 

            x = (n2-n1)/(m1 - m2)
            y = m1*x + n1 
        
        return ((x,y) == (x1,y1) and norma2((x,y), (x4,y4)) - norma2((x,y),(x2,y2))  <= 1) or (
                        (x,y) == (x3,y3) and norma2((x,y),(x2,y2)) - norma2((x,y), (x4,y4))  <= 1) or (
                            (x,y) != (x1,y1) and (x,y) != (x3,y3) and abs(norma2((x,y),(x2,y2)) - norma2((x,y), (x4,y4))) <= 1)

    
    def colinear_valid(self,p1, c1, p2, c2):
        f = lambda p1,c1,p2,c2 :(p1<= c1 and c1<p2 and c1< c2) or p1< p2< c1< c2  
        return  f(p1,c1,p2,c2) or f(p2,c2,p1,c1) or f(c1,p1,c2,p2) or f(c2,p2,c1,p1)

    

    
