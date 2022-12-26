import math
from entities.utils import DIRECTIONS, I_DIR, J_DIR, validMove

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


class FindVoronoiVertex(Problem):
    def __init__(self, roadmap, *args, **kwargs):
        self.roadmap = roadmap
        self.dim_x, self.dim_y = roadmap.distance.shape
        Problem.__init__(self, *args, **kwargs)

    def actions(self, state):
        """The actions executable in this state."""
        x, y = state
        dirs = [(dirvar.name, (I_DIR[dirvar.value - 1], J_DIR[dirvar.value - 1])) for dirvar in DIRECTIONS]
        return [(name, move) for name, move in dirs 
                                    if validMove(x + move[0], y + move[1], self.dim_x, self.dim_y) and 
                                       len(set(self.roadmap.color[x + move[0], y + move[1]]).intersection(self.roadmap.color[x, y])) > 0]

    def result(self, state, action):
        """The state that results from executing this action in this state."""
        x, y = state
        _, (x_dir, y_dir) = action
        return x + x_dir, y + y_dir

    def is_goal(self, state):
        """True if the goal level is in any one of the jugs."""
        return state in self.goals


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
        areas = self.roadmap.color[state]
        actions = []
        for i in range(0, len(areas)):
            for j in range(i + 1, len(areas)):
                min = areas[i]
                max = areas[j]
                if min > max:
                    (min, max) = (max, min)
                actions += self.roadmap.adjacents.get((min, max)) or []
        set_actions = set([node.state for node in actions]).difference([state])
        if state in self.start_end:
            set_actions.add(self.end_cost[state][-1])
        return set_actions

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