import sys, os
sys.path.append(os.path.abspath('.'))

from entities._units import Fighter
import entities.agent  as A
from entities.connector import StateMannager as SM


item = Fighter()
agent = A.Agent(A.RandomMove_IA())
state = SM(None, [(agent,[(item,(0,0)),(item,(0,1))])])
agent.move_connectors(None)
print(item.get_attack_range)