import numpy as np
import math
from collections import Counter

def winrate_to_score(winrate, player):
    if player == 1:
        return winrate*2 -1
    else:
        return (1-winrate)*2 -1

print(winrate_to_score(0.3, 1))
print(winrate_to_score(1, -1))