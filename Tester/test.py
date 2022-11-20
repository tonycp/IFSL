import sys, os
sys.path.append(os.path.abspath('.'))
from entities._units import Fighter

item = Fighter()
print(item.GetAttackRange)