import numpy as np

z = np.random.random((84, 1))
zT = z.T
zN = np.transpose(z)
print(z.shape)
print(zT.shape)