import sys, os
sys.path.append(os.path.abspath('.'))
from entities._units import Fighter

class test:
    __id = 0
    def __init__(self):
        test.__id += 1
        self.id = test.__id