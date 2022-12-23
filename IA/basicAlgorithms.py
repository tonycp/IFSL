from queue import PriorityQueue
from ._definitions import *

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