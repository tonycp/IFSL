# import sys, os
# sys.path.append(os.path.abspath('.'))
# import numpy as np
# from entities._units import Fighter
# import entities.agent  as A
# from entities.connector import StateMannager as SM
# from IA.basicAlgorithms import *
# import UI
# from entities._units import Knight


# # item = Fighter()
# # agent = A.Agent(A.RandomMove_IA())
# # state = SM(None, [(agent,[(item,(0,0)),(item,(0,1))])])
# # agent.move_connectors(None)
# # print(item.get_attack_range)


# roadmap = RoadMap(UI.examples.get_example_grid(UI.examples.obstacles), Knight())
# filter = lambda x, y: True
# is_empty = lambda x, y: roadmap.distance[x, y] != 0

# cells_area, cell_size, decomposed = slice((0, 0), roadmap.distance.shape[0], roadmap.distance.shape[1], filter, is_empty)
# cell_info = []
# for i in range(len(cells_area)):
#     cell_info.append(Cell_Info(cells_area[i],cell_size[i + 1]))
# distance = create_min_distance((0,0))
# best,best_rotation = GA(pob_generator= create_knuth_shuffle_pob_generator(len(cell_info),20),
#    parents_selector= RouletteWheel_ParentSelector,
#    crossover_operator= create_crossover_operator(create_crossover_cost(distance)),
#    mutation_operator= mutation_sequence_creator([swap_mutator,create_two_opt_mutator(distance,200)]),
#    fitness_func= create_fitness_func(distance),
#    cell_info = cell_info,
#    top_generation= 20
#     )
# print("Finish")


print([(1, *i) for i in [(1, 2), (2, 2)]])