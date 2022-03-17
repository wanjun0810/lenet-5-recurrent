import numpy as np
a = np.random.random(size=(120, 1))

print(a)
a = a.reshape([120, 1, 1])
print(a)