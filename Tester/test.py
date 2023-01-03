import sys, os
sys.path.append(os.path.abspath('.'))
import numpy as np
from entities._units import Fighter
import entities.agent  as A
from entities.connector import StateMannager as SM
from IA.basicAlgorithms import *


# item = Fighter()
# agent = A.Agent(A.RandomMove_IA())
# state = SM(None, [(agent,[(item,(0,0)),(item,(0,1))])])
# agent.move_connectors(None)
# print(item.get_attack_range)

cell_info = []
distance = create_min_distance((0,0))
GA(pob_generator= create_knuth_shuffle_pob_generator(len(cell_info),20),
   parents_selector= RouletteWheel_ParentSelector,
   crossover_operator= create_crossover_operator(create_crossover_cost(distance)),
   mutation_operator= mutation_sequence_creator([swap_mutator,create_two_opt_mutator(distance)]),
   fitness_func= create_fitness_func(distance),
   cell_info = cell_info,
   top_generation= 20
    )