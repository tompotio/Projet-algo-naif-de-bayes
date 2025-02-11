import numpy as np
import os
import math

import numpy as np

x = np.array([0, 0, 1, 1, 0, 1])
test = np.array([1, 1, 2, 3, 1, 1])

print(np.sum(test[x == 1]))
