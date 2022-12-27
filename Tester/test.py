import sys, os
sys.path.append(os.path.abspath('.'))
import numpy as np
from entities._units import Fighter
import entities.agent  as A
from entities.connector import StateMannager as SM


# item = Fighter()
# agent = A.Agent(A.RandomMove_IA())
# state = SM(None, [(agent,[(item,(0,0)),(item,(0,1))])])
# agent.move_connectors(None)
# print(item.get_attack_range)

state = np.ndarray(shape= 2, dtype=tuple)
state[0] = (1,2)
print(state)