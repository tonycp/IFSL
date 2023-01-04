from entities.connector import StateMannager as S
from IA.basicAlgorithms import *
from IA._definitions import *
from entities.utils import *
from random import randint
from math import inf


class RandomMove_IA(object):
    def __init__(self, map) -> None:
        self.map = map

    def get_move_for(self, connector: S.Connector):
        if(connector.state == STATES.Stand):
            dir = randint(0, 7)
            connector.notify_move(connector, int_to_direction(dir))
        elif(connector.timer > 0):
            connector.notify_move(connector, connector.prev_dir)


class Path_Move_IA(object):
    def __init__(self) -> None:
        self.goal = None

    def set_goal(self, goal=None):
        self.goal = goal
        self.path = []

    def get_move_for(self, poss):
        if len(self.goalpath) == 0:
            self.goal = None
            self.path = None
        if self.goal is None: return
        if not self.start:
            self.subgoal = self.goalpath.pop(0)
        self.steer_toward(poss)
        return self.path

    def steer_toward(self, poss):
        pass
 
    def next_path(self, position):
        raise Exception("not implement")

    def viewrange(self, poss, actposs, vision):
        return norma2(poss, actposs) <= vision

class SliceMapMove_IA(Path_Move_IA):
    def __init__(self) -> None:
        Path_Move_IA.__init__(self)
        self.dirs = [(I_DIR[dirvar.value - 1], J_DIR[dirvar.value - 1]) for dirvar in DIRECTIONS]

    def set_goal(self, start, goal=None, roadmap=None):
        Path_Move_IA.set_goal(self, goal)
        self.roadmap = roadmap
        if goal is not None:
            self.start = True
            self.position_start = start

            colors, size = self.goal = goal
            self.shape_x, self.shape_y = (x_max, x_min), (y_max, y_min) = size
            filter = lambda x, y: colors.intersection(roadmap.color[x, y])
            is_empty = lambda x, y: roadmap.distance[x, y] != 0
            cell_size, cell_floor, cell_ceiling, _ = slice((x_min, y_min), x_max - x_min + 1, y_max - y_min + 1, filter, is_empty)

            cell_info:list[Cell_Info] = []
            for size, ceiling, floor in zip(cell_size.values(), cell_floor.values(), cell_ceiling.values()):
                cell_info.append(Cell_Info(size, ceiling, floor))

            distance = create_min_distance(start)
            best, best_rotation = GA(
                pob_generator= create_knuth_shuffle_pob_generator(len(cell_info),20),
                parents_selector= RouletteWheel_ParentSelector,
                crossover_operator= create_crossover_operator(create_crossover_cost(distance)),
                mutation_operator= mutation_sequence_creator([swap_mutator,create_two_opt_mutator(distance,200)]),
                fitness_func= create_fitness_func(distance),
                cell_info = cell_info,
                top_generation= 20
                )

            self.cell_info = cell_info
            self.goalpath = list(zip(best, best_rotation))
            self.subgoal = None
            self.change = False

    def viewrange(self, next, actposs, vision):
        i, r = next
        info_path = self.cell_info[i].rotations[r]
        return Path_Move_IA.viewrange(self, info_path.start, actposs, vision)

    def steer_toward(self, poss):
        i, r = self.goalpath[0] if self.subgoal is None and self.goal is not None else self.subgoal
        info_path = self.cell_info[i].rotations[r]

        if self.start:
            self.start = False
            self.path = self.solve_initial(info_path.start)
        elif poss == info_path.start:
            x, y = poss
            self.path = []
            count = 0
            while count < len(info_path.path):
                dx, dy = info_path.path[count]
                if x == dx and y == dy:
                    count += 1
                    continue
                increment = -1 if x > dx or y > dy else 1
                for i in range(x + increment, dx + increment, increment):
                    self.path.append((i, y))
                for i in range(y + increment, dy + increment, increment):
                    self.path.append((x, i))
                x, y = dx, dy
                count += 1
            if not len(self.goalpath):
                return
            i, r = self.goalpath[0]
            info_path_go = self.cell_info[i].rotations[r]
            astar_path = self.next_path(info_path.end, info_path_go.start)
            self.path += astar_path

    def next_path(self, position, end):
        firter = lambda x: x.state == end
        adj = lambda x: [NodeTree(state=(x.state[0] + dir_x, x.state[1] + dir_y), parent=x, path_cost=x.path_cost + 1) 
                            for dir_x, dir_y in self.dirs 
                                if util.validMove(x.state[0] + dir_x, x.state[1] + dir_y, self.shape_x[0] + 1, self.shape_y[0] + 1, self.shape_x[1], self.shape_y[1]) and 
                                    self.roadmap.distance[x.state[0] + dir_x, x.state[1] + dir_y] != 0]
        he = lambda x: norma2(x.state, end)
        new_path = path_states(astar_search(NodeTree(state=position), firter, adj, he))[1:]
        return new_path
    
    def solve_initial(self, end):
        firter = lambda x: x.state == end
        adj = lambda x: [NodeTree(state=(x.state[0] + dir_x, x.state[1] + dir_y), parent=x, path_cost=x.path_cost + 1) 
                            for dir_x, dir_y in self.dirs 
                                if util.validMove(x.state[0] + dir_x, x.state[1] + dir_y, *self.roadmap.distance.shape) and 
                                    self.roadmap.distance[x.state[0] + dir_x, x.state[1] + dir_y] != 0]
        he = lambda x: norma2(x.state, end)
        new_path = path_states(astar_search(NodeTree(state=self.position_start), firter, adj, he))[1:]
        return new_path


class RoadMapMove_IA(Path_Move_IA):
    def set_goal(self, start, goal=None, roadmap=None):
        Path_Move_IA.set_goal(self, goal)
        self.roadmap = roadmap
        if goal is not None:
            self.start = True
            position_start = start
            vertexs_state_start = self.roadmap.get_vertex_area(*position_start)
            findproblem_start = FindVoronoiVertex(self.roadmap, vertexs_state_start)

            start = NodeTree(state=position_start)
            self.start_costs = dict([(node.state, path_states(node)[1:]) 
                                 for node in breadth_first_search_problem(start, findproblem_start)])

            position_end = self.goal
            vertexs_state_end = self.roadmap.get_vertex_area(*position_end)
            findproblem_end = FindVoronoiVertex(self.roadmap, vertexs_state_end)

            end = NodeTree(state=goal)
            self.end_cost = dict([(node.state, path_states(node)[-2::-1])
                                 for node in breadth_first_search_problem(end, findproblem_end)])
            start_end = [state for state in self.end_cost]
            vertexproblem = MoveVoronoiProblem(
                start_costs=self.start_costs, end_cost=self.end_cost, start_end=start_end, roadmap=self.roadmap, goals=[goal])
            self.goalpath = path_states(
                astar_search_problem(start, vertexproblem,
                                     lambda n: norma2(n.state, end.state)))[1:]
    
    def steer_toward(self, poss):
        if len(self.roadmap.color[poss]) >= 3:
            self.path = self.next_path(poss)
        elif self.start:
            self.path = self.start_costs[self.goalpath[0]]
            self.start = False
        elif self.goal == self.subgoal:
            self.path = self.end_cost[self.subgoal]
    
    def next_path(self, position):
        colors = set(self.roadmap.color[position]).intersection(self.roadmap.color[self.subgoal])

        findproblem_end = FindVoronoiVertex(self.roadmap, [self.subgoal])
        end = NodeTree(state=position)
        filter = lambda n: findproblem_end.is_goal(n.state)
        newexpand = lambda n: [] if len(colors.intersection(self.roadmap.color[n.state])) != 2 else expand(problem=findproblem_end, node=n)
        path = path_states(best_first_search(end, filter, newexpand))[1:]
        return path