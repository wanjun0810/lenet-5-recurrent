import numpy as np

# z = np.random.random((2 , 2))*100
z = [[2,8],[4,6]]
k = np.ones((2 , 2)) * (1 / 4)
x = np.kron(z, k)
print(z)
print('---------------------------')
print(x)