from random import randint
import numpy as np
import math
from entities.utils import DIRECTIONS, I_DIR, J_DIR, norma_inf, validMove, norma2, validMove, int_to_direction
from collections import defaultdict


#*####################################################################################
#*                                     Search                                        #
#*####################################################################################

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

            asignment = dict([(i,j) for i,j in copy_asignment.items()])
            self.domain = dict([(i,j.copy()) for i,j in domain_copy.items()])
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
    
    def segment_interception(self,x1, x2, x3, x4, y1, y2, y3, y4):
        
        if not CSP_UncrossedAsignment.segment_interception(self, x1, x2, x3, x4, y1, y2, y3, y4): return False

        #si el origen de cada conector es igual a su destino, son dos puntos y no se cruzan
        if (x1,y1)==(x2,y2) and (x3,y3) == (x4,y4): return False
        elif (x1,y1)==(x2,y2) or (x3,y3) == (x4,y4): #Si al menos un conector esta n su destino basa con ver que no este en medio del camino del otro
            return not (CSP_UncrossedAsignmentTime.colinear_valid(x1,x2,x3,x4) or CSP_UncrossedAsignmentTime.colinear_valid(y1,y2,y3,y4))


        if x1==x2 and x3==x4: #Si son dos rectas veticales se busca condicion de cruce en una linea en el plano rotado
            return not CSP_UncrossedAsignmentTime.colinear_valid(y1,y2,y3,y4)
        
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
                return not CSP_UncrossedAsignmentTime.colinear_valid(x1, x2, x3, x4)
            
            n1 = y1 - m1*x1
            n2 = y3 - m2*x3 

            x = (n2-n1)/(m1 - m2)
            y = m1*x + n1 
        
        return ((x,y) == (x1,y1) and norma2((x,y), (x4,y4)) - norma2((x,y),(x2,y2))  <= 1) or (
                        (x,y) == (x3,y3) and norma2((x,y),(x2,y2)) - norma2((x,y), (x4,y4))  <= 1) or (
                            (x,y) != (x1,y1) and (x,y) != (x3,y3) and abs(norma2((x,y),(x2,y2)) - norma2((x,y), (x4,y4))) <= 1)

    
    def colinear_valid(p1, c1, p2, c2):
        f = lambda p1,c1,p2,c2 :(p1<= c1 and c1<=p2 and c1< c2) or p1< p2< c1< c2  
        return  f(p1,c1,p2,c2) or f(p2,c2,p1,c1) or f(c1,p1,c2,p2) or f(c2,p2,c1,p1)

 
def evaluate_HillClimbingAsignment(csp, connectors, goals, alpha = 1):
    cost = np.ndarray(shape = len(goals), dtype=float)
    for i in range(0,len(goals)):
        (x1,y1) = goals[i]
        (x2,y2) = connectors[i].get_position()
        cost[i] += norma2((x1,y1),(x2,y2))
        for j in range(i,len(goals)):
            (x3,y3) = goals[i]
            (x4,y4) = connectors[i].get_position()
            if CSP_UncrossedAsignmentTime.segment_interception(csp, x1,x2,x3,x4,y1,y2,y3,y4):
                cost[i]+=1 * alpha
                cost[j]+=1 * alpha
    return max(cost)
            


    
#!###############################################################################################
#!                                            MiniMax                                           #
#!###############################################################################################  

infinity = math.inf
  
class Game:

    def actions(self, state):
        """Return a collection of the allowable moves from this state."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def is_terminal(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)
    
    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError
    
    def undo(self, state, action):raise NotImplementedError
        
class FigthGame(Game):
    
    def actions(self, state):
        player = state.to_move
        state.p = state.p1
        player2 = state.p2
        if player == player2["id"]:
            player1, player2 = player2,player1
        actions =[]
        x,y = player1["pos"][-1]
        x2,y2 = player2["pos"][-1]
        if norma_inf((x,y),(x2,y2)) <= player1["attack"]:
            actions.append(("attack", (x2, y2)))
        if player1["move"] != infinity:
            for z in range(0,len(I_DIR)):
                i = I_DIR[z]
                j = J_DIR[z]
                if((x + i, y + j) not in state.bussy and validMove(x + i, y + j, state.map.shape[0], state.map.shape[1]) and (state.map[x + i, y + j].is_empty or player1["pos"][0] == (x + i, y + j))):
                        actions.append(("move", (x + i, y + j)))
        actions.append(("wait", (x, y)))
        return actions
    
    def result(self, state, move):
        player = state.to_move
        player1 = state.p1
        player2 = state.p2
        if player == player2["id"]:
            player1, player2 = player2, player1
        
        action, direction = move 
        if action == "move":
            player1["pos"].append(direction)
        elif action == "attack":
            if player1["move"] == infinity or player2["move"] == infinity:
                player2["hp"].append(player2["hp"][-1] - player1["dmg"])
            else:
                times = 0
                for _ in range(0,200):
                    if randint(1, player1["move"] + player2["move"]) <= player2["move"]:
                        times+=1
                player2["hp"].append(player2["hp"][-1] - player1["dmg"] * times/200)
        state.to_move = player2["id"]
        return state
            
    def is_terminal(self, state):
        return state.p1["hp"][-1] <= 0 or state.p2["hp"][-1] <= 0 
    
    def utility(self, state, player):
        if self.is_terminal(state):
            if state.p1["hp"][-1] <= 0 and state.p1["id"] == player:
                return -1000
            return 1000
        return 0
    
    def heuristic(self, state, player):
        player1 = state.p1
        player2 = state.p2
        if player == player2["id"]:
            player1, player2 = player2,player1
        
        #bonificar la reduccion de vida del oponente
        h = player2["hp"][0] - player2["hp"][-1]
        
        #penalizar valor si pierdes al oponente de vista
        if norma_inf(player1["pos"][-1], player2["pos"][-1]) > player1["view"]:
            h -= 10
         
        #penalizar si se va de tu rango de ataque   
        if norma_inf(player1["pos"][-1], player2["pos"][-1]) > player1["attack"]:
            h -= 5
        
        #bonificar si tienes al oponente dentro de tu rango de ataque pero el no te puede atacar a ti
        if player1["attack"] >= norma_inf(player1["pos"][-1], player2["pos"][-1]) >= player2["attack"]:
            h += 10
        
        #bonificar si el oponente esta en tu rango de ataque
        if norma_inf(player1["pos"][-1], player2["pos"][-1]) <= player1["attack"]:
            h += 5

        return h
         
    def undo(self, action, state):
        player = state.to_move
        player1 = state.p1
        player2 = state.p2
        if player == player1["id"]:
            player1, player2 = player2, player1
        name, _ = action
        if  name == "attack":
            player2["hp"].pop()
        elif name == "move":
            player1["pos"].pop()
        state.to_move = player1["id"]
                     
class State:
    
    def __init__(self, map, connector1, connector2, bussy =None, **kwds):
        self.p1 ={} 
        self.p1["id"] = connector1.id
        self.p1["attack"] = connector1.unit.get_attack_range
        self.p1["hp"] = [connector1.get_health_points()]
        self.p1["view"] = connector1.unit.get_vision_radio
        self.p1["dmg"] = connector1.unit.get_damage
        self.p1["move"] = connector1.unit.get_move_cost
        self.p1["pos"] = [connector1.get_position()] 
        
        
        self.p2 ={} 
        self.p2["id"] = connector2.id
        self.p2["attack"] = connector2.unit.get_attack_range
        self.p2["hp"] = [connector2.get_health_points()]
        self.p2["view"] = connector2.unit.get_vision_radio
        self.p2["dmg"] = connector2.unit.get_damage
        self.p2["move"] = connector2.unit.get_move_cost
        self.p2["pos"] = [connector2.get_position()] 
        
        self.to_move = connector1.id
        self.bussy =bussy
        self.map = map

class Group_State:

    def __init__(self, map, connector, alliads, enemies, attacks, bussy=None, **kwds):
        self.alliads_attacks = np.ndarray(shape=len(alliads), dtype=int) 
        self.alliads_moves = np.ndarray(shape=len(alliads), dtype=int) 
        self.enemies_attacks = np.ndarray(shape=len(enemies), dtype=int) 
        self.attacks = attacks
        self.enemies_pos_and_range =np.ndarray(shape=len(alliads)) 
        acumulator= 0
        
        for j in range(len(enemies)):
            
            en = enemies[j]
            acumulator += en.get_health_points()
            self.enemies_pos_and_range[j] = (en.get_position(), en.unit.get_attack_range, en.get_health_points(), en.unit.get_damage)
            
            for i in range(len(alliads)):
                a= alliads[i]
                distance = norma_inf(a.get_position(), en.get_position())

                if distance <=  a.unit.get_attack_range: 
                    self.alliads_attacks[i]+=1
                if distance <=  en.unit.get_attack_range: 
                    self.enemies_attacks[j]+=1
        
        self.enemies_total_live= [acumulator]
        
        self.alliads_pos_and_range =np.ndarray(shape=len(alliads))
        acumulator = 0
        for n in range(len(alliads)):
            a= alliads[x]
            acumulator += a.get_health_points()
            self.alliads_pos_and_range[n] = a.unit.get_damage
            x,y = a.get_position()
            for z in range(0,len(I_DIR)):
                    i = I_DIR[z]
                    j = J_DIR[z]
                    if validMove(x + i, y + j, map.shape[0],map.shape[1]) and map[x + i, y + j].is_empty:
                        self.alliads_moves[i] +=1

        self.p ={} 
        self.p["attack"] = connector.unit.get_attack_range
        self.p["hp"] = [connector.get_health_points()]
        self.p["view"] = connector.unit.get_vision_radio
        self.p["dmg"] = connector.unit.get_damage
        self.p["move"] = connector.unit.get_move_cost
        self.p["pos"] = [connector.get_position()] 
        
        self.to_move = "warrior"
        self.bussy =bussy
        self.map = map

class GroupFigthGame(Game):
    
    def is_terminal(self, state ):
        return state.p["hp"][-1] <= 0 or state.enemies_total_live[-1] <=0  
    
    def utility(self, state, player):
        if self.is_terminal(state):
            if player== "warrior" and state.enemies_total_live[-1] <=0:
                return 10000
            return -10000
        return 0
    
    def actions(self, state):
        player = state.to_move
        actions =[]
        if player == "warrior":
            player1= state.p
            x,y = player1["pos"][-1]
            hbest = infinity
            pbest=None
            for (x2,y2), _ ,h in state.enemies_pos_and_range:
                if norma_inf((x,y),(x2,y2)) <= player1["attack"] and h < hbest:
                    pbest = (x2,y2)
                    hbest =h 
            if pbest:
                actions.append(("attack", pbest))
            for z in range(0,len(I_DIR)):
                i = I_DIR[z]
                j = J_DIR[z]
                if((x + i, y + j) not in state.bussy and validMove(x + i, y + j, state.map.shape[0], state.map.shape[1]) and (state.map[x + i, y + j].is_empty or player1["pos"][0] == (x + i, y + j))):
                        actions.append(("move", (x + i, y + j)))
            actions.append(("wait", (x, y)))
        else: 
           actions.append(("tower", (x, y))) 
        return actions

    def result(self, state, move):
        action, direction = move 
        if action != "tower":
            state.p["pos"].append(direction)
            total_attack = 0
            for i in range(len(state.alliads_moves)):
                prob =state.alliads_attacks[i]/(state.alliads_moves[i] + state.alliads_attacks[i]+1)
                dmg = state.alliads_pos_and_range[i]
                if randint(0,100)/100 < prob:
                    total_attack += dmg
            state.enemies_total_live.append(state.enemies_total_live[-1]- total_attack)

            if action == "move":
                state.p["pos"].append(direction)
            else:
                state.enemies_total_live[-1]-= state.p["dmg"]
        
        else: 
            total_attack = 0
            for i in range(len(state.enemies_attacks)):
                pos, ran, _ , dmg= state.enemies_pos_and_range[i]
                prob = (1 if norma_inf(state.p["pos"][-1], pos) <= ran else 0)/(state.enemies_attacks[i] + 1)
                if prob > 0 and randint(0,100)/100 < prob:
                    total_attack += dmg
            state.p["hp"].append(state.p["hp"][-1]- total_attack )


        state.to_move = "warrior" if state.to_move == "towers" else "towers"
        return state

    def undo(self, action, state):
        name, _ = action
        if  name == "attack":
            state.enemies_total_live["hp"].pop()
        elif name == "move":
            state.enemies_total_live["hp"].pop()
            state.p["pos"].pop()
        else: 
            state.p["hp"].pop()
        state.to_move = "warrior" if name == "tower" else "tower"

    def heuristic(self, state: Group_State, player):

        if player == "warrior":
            #bonificar la reduccion de vida del oponente
            h = state.enemies_total_live[0] - state.enemies_total_live[-1]


            for pos,rng,_,_ in state.enemies_pos_and_range: 
                
                #bonificar si el oponente esta en tu rango de ataque
                if norma_inf(state.p["pos"][-1], pos) <= state.p["attack"]:
                    h += 5
                
                #bonificar si tienes al oponente dentro de tu rango de ataque pero el no te puede atacar a ti
                if state.p["attack"] >= norma_inf(state.p["pos"][-1], pos) >= rng:
                    h += 10
                
                #penalizar valor si pierdes al oponente de vista
                if norma_inf(state.p["pos"][-1], pos) > state.p["view"]:
                    h -= 10

                #penalizar si se va de tu rango de ataque   
                if norma_inf(state.p["pos"][-1], pos) > state.p["attack"]:
                    h -= 5
        else: 
            h = state.p["hp"][0] - state.p["hp"][-1]
        return h



def h_alphabeta_search_solution(game, state, cutoff, h):
    player = state.to_move

    def max_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta, depth+1)
            game.undo(a,state)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    def min_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta, depth + 1)
            game.undo(a,state)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -infinity, +infinity, 0)
