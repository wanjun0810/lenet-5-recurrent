import numpy as np


def convolution_3d(data, kernel):
    data_num, data_row, data_col = data.shape
    kernel_num, kernel_row, kernel_col = kernel.shape
    n = data_col - kernel_col
    m = data_row - kernel_row
    state = np.zeros((data_num, m + 1, n + 1))
    for i in range(data_num):
        for j in range(m + 1):
            for k in range(n + 1):
                temp = np.multiply(
                    data[i][j:j + kernel_row, k:k + kernel_col], kernel[i])
                state[i][j][k] = temp.sum()
    return state


if __name__ == "__main__":
    a = np.random.random(size=(3, 3, 3))
    b = np.ones((3, 3, 3)) 
    z = convolution_3d(a, b)
    print(np.sum(a[1]))
    print(a)
    print(b)
    print(z)
    

