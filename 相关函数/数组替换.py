import numpy as np

a = np.zeros((5, 5,5))
a[a == 0] = 1
print(a)