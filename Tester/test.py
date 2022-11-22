import sys, os
sys.path.append(os.path.abspath('.'))
from entities._units import Fighter
from enum import Enum

item = Fighter()
print(item.GetAttackRange)