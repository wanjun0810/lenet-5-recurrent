import numpy as np
import math
import struct
import time

# 训练集文件
train_images_idx3_ubyte_file = 'mnist/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'mnist/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 'mnist/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'mnist/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):

    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)
    # 在代码中出现但没有解释的数字常量或字符串称为魔数 (magic number)或魔字符串。
    # 根据这几个字节的内容就可以确定文件类型
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' %
          (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    offset += struct.calcsize(fmt_header)
    print(offset)
    # 图像数据像素值的类型为unsigned char型，对应的format格式为B。
    # 这里还有加上图像大小784，是为了读取784个B格式数据，
    # 如果没有则只会读取一个值（即一副图像中的一个像素值）
    fmt_image = '>' + str(image_size) + 'B'
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(
            fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # print(images[i])
        offset += struct.calcsize(fmt_image)
    # plt.imshow(images[i],'gray')
    # plt.pause(0.00001)
    # plt.show()
    # plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


# 矩阵相关操作
def convolution(data, kernel):
    # data_row 要进行相关操作的特征图的行数
    # data_col 要进行相关操作的特征图的列数
    # shape函数是numpy.core.fromnumeric中的函数，查看矩阵维数及长度。
    data_row, data_col = data.shape

    # kernel_row 要进行相关操作的卷积核的行数
    # kernel_col 要进行相关操作的卷积核的列数
    kernel_row, kernel_col = kernel.shape

    # 输出大小等于输入大小减去滤波器大小加上1，最后再除以步长
    # m 相关结果的行数  eg: 32 - 5 + 1 = 28
    m = data_row - kernel_row + 1
    # n 相关结果的列数
    n = data_col - kernel_col + 1
    # state 相关结果矩阵 大小(n, m)
    state = np.zeros((m, n))

    # 循环生成结果矩阵内容
    for i in range(m):
        for j in range(n):
            # temp 数组和矩阵对应位置相乘(np.multiply)，输出与相乘矩阵(kernel)的大小一致
            temp = np.multiply(
                data[i:i + kernel_row, j:j + kernel_col], kernel)
            # 相关结果输出等于对应位相乘结果的和
            state[i][j] = temp.sum()

    return state


# Relu激活函数 三维
def relu(feature_map):
    # feature_map_num 要激活的特征图的数量
    # feature_map_row 要激活的特征图的行数
    # feature_map_col 要激活的特征图的列数
    # shape函数是numpy.core.fromnumeric中的函数，查看矩阵维数及长度。
    feature_map_num, feature_map_row, feature_map_col = feature_map.shape

    # feature_map_relu 保存激活后的特征图
    feature_map_relu = np.zeros(
        (feature_map_num, feature_map_row, feature_map_col))

    # 激活函数实现
    for i in range(0, feature_map_num):
        for j in range(0, feature_map_col):
            for k in range(0, feature_map_row):
                # 如果输入数据大于0保留，如果输入数据小于0置为0
                if feature_map[i][j][k] > 0:
                    feature_map_relu[i][j][k] = feature_map[i][j][k]
                else:
                    feature_map_relu[i][j][k] = 0

    return feature_map_relu


# 激活函数 二维  84*1 10*1
def relu_2d(feature):
    # feature_map_row 要激活的特征图的行数
    # feature_map_col 要激活的特征图的列数
    # shape函数是numpy.core.fromnumeric中的函数，查看矩阵维数及长度。
    feature_row, feature_col = feature.shape
    # feature_map_relu 保存激活后的特征图
    feature_relu = np.zeros((feature_row, feature_col))

    # 激活函数实现
    for i in range(0, feature_row):
        for j in range(0, feature_col):
            # 如果输入数据大于0保留，如果输入数据小于0置为0
            if feature[i][j] > 0:
                feature_relu[i][j] = feature[i][j]
            else:
                feature_relu[i][j] = 0

    return feature_relu


# 池化操作(平均池化 + 池化层偏置)
def pool(feature_map, poolSize, pool_b):
    # feature_map_num 要池化的特征图的数量
    # feature_map_row 要池化的特征图的行数
    # feature_map_col 要池化的特征图的列数
    # shape函数是numpy.core.fromnumeric中的函数，查看矩阵维数及长度。
    feature_map_num, feature_map_col, feature_map_row = feature_map.shape

    # w_Len 池化结果的行数  h_Len 池化结果的列数
    # 输出大小等于输入大小减去滤波器大小除以步长加上1
    w_Len = int((feature_map_col-poolSize)/poolSize+1)
    h_Len = int((feature_map_row - poolSize) / poolSize+1)
    # poolResult 存储池化后的结果
    poolResult = np.zeros((feature_map_num, w_Len, h_Len))

    # 池化操作
    for i in range(0, feature_map_num):
        for j in range(0, w_Len):
            for k in range(0, h_Len):
                # poolField 依次提取第i张特征图要池化的特征图范围(2 * 2)
                poolField = feature_map[i][j*poolSize:j*poolSize +
                                           poolSize, k * poolSize: k * poolSize + poolSize]
                # 平均池化 取特征图范围的平均值
                poolResult[i][j][k] = np.mean(poolField)
        # 为池化结果添加偏置
        poolResult[i] = poolResult[i] + pool_b[i]

    return poolResult


# softmax函数 求结果的概率 确定计算结果
def softmax(output):
    # size 获得output结点矩阵的大小
    size = output.shape
    # node output结点个数(size[0])
    node = size[0]

    # output_max 获得output结点中的最大值
    output_max = np.max(output)
    # softmax_result 存储softmax结果
    softmax_result = np.zeros((node, 1))
    '''
    # softmax可能会出现上溢出或下溢出情况
    # 解决方法：
    # 令output_max = max(output_i), i =1,2,3,….,n
    # 即output_max为所有output_i中最大的值
    # 则将计算softmax(output_i)的值转化为计算softmax(output_i - output_max)的值
    '''
    # safe 存储output_i - output_max的值
    safe = np.zeros((node, 1))
    safe = output - output_max  # 计算safe
    # denominator 分母  保存求和结果
    denominator = 0

    # 计算softmax函数使用的分母
    for i in range(0, node):
        denominator = denominator + np.exp(safe[i])
    # 实现softmax函数
    for i in range(0, node):
        softmax_result[i] = np.exp(safe[i]) / denominator

    return softmax_result


def relu_back_2d(feature):
    # feature_map_row 要池化的特征图的行数
    # feature_map_col 要池化的特征图的列数
    # shape函数是numpy.core.fromnumeric中的函数，查看矩阵维数及长度。
    feature_row, feature_col = feature.shape
    # feature_relu_back 存储输入的激活后的特征图的导数
    feature_relu_back = np.zeros((feature_row, feature_col))

    # 求特征图的导数
    for i in range(0, feature_row):
        for j in range(feature_col):
            # 如果激活后的特征图大于0，则其导数为1
            # 如果激活后的特征图等于0，则其导数为0
            if feature[i][j] > 0:
                feature_relu_back[i][j] = 1
            else:
                feature_relu_back[i][j] = 0

    return feature_relu_back


# 激活函数 三维
def relu_back(feature_map):
    # feature_map_num 要池化的特征图的数量
    # feature_map_row 要池化的特征图的行数
    # feature_map_col 要池化的特征图的列数
    # shape函数是numpy.core.fromnumeric中的函数，查看矩阵维数及长度。
    feature_map_num, feature_map_row, feature_map_col = feature_map.shape
    # feature_relu_back 存储输入的激活后的特征图的导数
    feature_map_relu_back = np.zeros(
        (feature_map_num, feature_map_row, feature_map_col))

    # 求特征图的导数
    for i in range(0, feature_map_num):
        for j in range(0, feature_map_col):
            for k in range(0, feature_map_row):
                # 如果激活后的特征图大于0，则其导数为1
                # 如果激活后的特征图等于0，则其导数为0
                if feature_map[i][j][k] > 0:
                    feature_map_relu_back[i][j][k] = 1
                else:
                    feature_map_relu_back[i][j][k] = 0
    return feature_map_relu_back


def lenet5():
    # 获取图片
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    start = time.time()

    # 卷积核及各卷积核偏置
    '''
    conv1 32 * 32 -> 28 * 28
    pool2 28 * 8 -> 14 * 14
    conv3 14 * 14 -> 10 * 10
    pool4 10 * 10 -> 5 * 5
    conv5 5 * 5 -> 1 * 1
    '''
    # He初始化
    # random.uniform的函数原型为：random.uniform(a, b)
    # 用于生成一个指定范围内的随机符点数，两个参数其中一个是上限，一个是下限。
    # 适用于ReLU的初始化方法 [0,sqrt(2/row*col*num)]

    # 计算第一层卷积的参数
    # 计算卷积层参数初始值上下限的绝对值
    conv1_init = math.sqrt(6 / (5 * 5 * 6))
    # 存储卷积层卷积核初始值
    conv1 = np.random.uniform(-conv1_init, conv1_init, size=(6, 5, 5))
    # 存储卷积层偏置初始值上下限的绝对值
    conv1_b = np.random.uniform(0, conv1_init, size=(6, 1))

    # 计算的第二层池化层的参数
    # 存储池化层偏置初始值上下限的绝对值
    pool2_init = math.sqrt(6 / (2 * 2))
    # 存储池化层偏置
    pool2_b = np.random.uniform(0, pool2_init, size=(6, 1))

    # 计算第三层卷积的参数
    # 计算卷积层参数初始值上下限的绝对值
    conv3_init = math.sqrt(6 / (5 * 5 * 16))
    # 存储卷积层卷积核初始值
    conv3 = np.random.uniform(-conv3_init, conv3_init, size=(16, 5, 5))
    # 存储卷积层偏置初始值
    conv3_b = np.random.uniform(0, conv3_init, size=(16, 1))

    # 计算第四层池化层的参数
    # 存储池化层偏置初始值上下限的绝对值
    pool4_init = math.sqrt(6 / (2 * 2))
    # 存储池化层偏置
    pool4_b = np.random.uniform(0, pool4_init, size=(16, 1))

    # 计算第五层卷积的参数
    # 计算卷积层参数初始值上下限的绝对值
    conv5_init = math.sqrt(6 / (5 * 5 * 120))
    # 存储卷积层卷积核初始值
    conv5 = np.random.uniform(-conv5_init, conv5_init, size=(120, 5, 5))
    # 存储卷积层偏置初始值
    conv5_b = np.random.uniform(0, conv5_init, size=(120, 1))
    # np.random.randn

    # 全连接层权值及偏置
    '''
    hide 120 -> 84
    output 84 -> 10
    '''
    # 计算隐藏层参数
    # 计算隐藏层参数初始值上下限的绝对值
    hide_init = math.sqrt(6 / (120 * 84))
    # 存储隐藏层权值初始值
    hide_w = np.random.uniform(-hide_init, hide_init, size=(84, 120))
    # 存储隐藏层偏置初始值
    hide_b = np.random.uniform(0, hide_init, size=(84, 1))

    # 计算输出层参数
    # 计算输出层参数初始值上下限的绝对值
    output_init = math.sqrt(6 / (84 * 10))
    # 存储输出层权值初始值
    output_w = np.random.uniform(-output_init, output_init, size=(10, 84))
    # 存储输出层偏置初始值
    output_b = np.random.uniform(0, output_init, size=(10, 1))

    # 学习率
    learn_rate = 1e-3  # 学习率
    # 分类结果
    # results = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

    bitchSize = 100  # 每bitchSize张图片更新一次参数
    # M = 0
    # 训练图片 M+N*bitchSize = 60000 -> M= 0
    # （for i in range(M ,M+bitchSize):pass \n if M+bitchSize  ...:M = 0 \n else: M = M+bitchSize）？
    learn_rate_change = np.zeros(10001)  # 若Loss变化率小于某个值时，学习率减小,根据学习次数设置
    loss = 0  # 整体误差
    exper = 0  # 测试次数
    # while (loss > 0.1):
    while (exper < 10000):
        '''
        MiniBitch 误差更新 
        loss累加
        其余参数累加，最后学习 
        '''
        loss = 0  # 整体误差
        output_b_loss = np.zeros((10, 1))  # 存储输出层偏置梯度
        output_w_loss = np.zeros((10, 84))  # 存储输出层权值梯度
        hide_b_loss = np.zeros((84, 1))  # 存储隐藏层偏置梯度
        hide_w_loss = np.zeros((84, 120))  # 存储隐藏层权值梯度
        conv5_b_loss = np.zeros((120, 1))  # 存储第五层偏置梯度
        conv5_loss = np.zeros((120, 5, 5))  # 存储第五层卷积核梯度
        pool4_b_loss = np.zeros((16, 1))  # 存储第四层偏置梯度
        conv3_b_loss = np.zeros((16, 1))  # 存储第三层参曾偏置梯度
        conv3_loss = np.zeros((16, 5, 5))  # 存储第三层卷积核梯度
        pool2_b_loss = np.zeros((6, 1))  # 存储第二层偏置梯度
        conv1_b_loss = np.zeros((6, 1))  # 存储第一层偏置梯度
        conv1_loss = np.zeros((6, 5, 5))  # 存储第一层卷积核梯度

        # 正向传播
        for i in range(bitchSize):
            train_image = np.pad(
                train_images[i], 2, 'constant')  # 将28*28的图片扩展为32*32 train_image为扩充后的图片
            # print(train_labels[i])
            '''
            卷积层C1
            '''
            feature_map1 = np.zeros((6, 28, 28))  # 存储第一层卷积结果
            for k in range(0, 6):
                feature_map1[k] = convolution(
                    train_image, conv1[k]) + conv1_b[k]  # 输入图片分别与六个卷积核相关 添加偏置
            feature_map1_relu = relu(feature_map1)  # 激活函数
            '''
            池化层P2
            pool:池化函数，图片大小减半添加偏置
            '''
            feature_map2 = pool(feature_map1_relu, 2,
                                pool2_b)  # 池化操作 图片大小减半添加偏置
            feature_map2_relu = relu(feature_map2)  # 激活函数
            '''
            卷积层C3
            十六个组合与十六个卷积核一对一相关
            '''
            feature_map3_0 = np.zeros((16, 14, 14))  # 第三层特征图的十六种组合的输入
            feature_map3_0[0] = feature_map2_relu[0] + \
                feature_map2_relu[1] + feature_map2_relu[2]
            feature_map3_0[1] = feature_map2_relu[1] + \
                feature_map2_relu[2] + feature_map2_relu[3]
            feature_map3_0[2] = feature_map2_relu[2] + \
                feature_map2_relu[3] + feature_map2_relu[4]
            feature_map3_0[3] = feature_map2_relu[3] + \
                feature_map2_relu[4] + feature_map2_relu[5]
            feature_map3_0[4] = feature_map2_relu[0] + \
                feature_map2_relu[4] + feature_map2_relu[5]
            feature_map3_0[5] = feature_map2_relu[0] + \
                feature_map2_relu[1] + feature_map2_relu[5]
            feature_map3_0[6] = feature_map2_relu[0] + feature_map2_relu[1] + \
                feature_map2_relu[2] + feature_map2_relu[3]
            feature_map3_0[7] = feature_map2_relu[1] + feature_map2_relu[2] + \
                feature_map2_relu[3] + feature_map2_relu[4]
            feature_map3_0[8] = feature_map2_relu[2] + feature_map2_relu[3] + \
                feature_map2_relu[4] + feature_map2_relu[5]
            feature_map3_0[9] = feature_map2_relu[0] + feature_map2_relu[3] + \
                feature_map2_relu[4] + feature_map2_relu[5]
            feature_map3_0[10] = feature_map2_relu[0] + \
                feature_map2_relu[1] + \
                feature_map2_relu[4] + feature_map2_relu[5]
            feature_map3_0[11] = feature_map2_relu[0] + \
                feature_map2_relu[1] + \
                feature_map2_relu[2] + feature_map2_relu[5]
            feature_map3_0[12] = feature_map2_relu[0] + \
                feature_map2_relu[1] + \
                feature_map2_relu[3] + feature_map2_relu[4]
            feature_map3_0[13] = feature_map2_relu[1] + \
                feature_map2_relu[2] + \
                feature_map2_relu[4] + feature_map2_relu[5]
            feature_map3_0[14] = feature_map2_relu[0] + \
                feature_map2_relu[2] + \
                feature_map2_relu[3] + feature_map2_relu[5]
            feature_map3_0[15] = feature_map2_relu[0] + feature_map2_relu[1] + \
                feature_map2_relu[2] + feature_map2_relu[3] + \
                feature_map2_relu[4] + feature_map2_relu[5]
            # 输入特征图求和平均
#            for j in range(0,6):
#                feature_map3_0[j] = feature_map3_0[j] / 3
#            for j in range(6, 15):
#                feature_map3_0[j] = feature_map3_0[j] / 4
#            feature_map3_0[15] = feature_map3_0 / 6
            feature_map3 = np.zeros((16, 10, 10))  # 存储第三层特征图
            for j in range(16):
                feature_map3[j] = convolution(
                    feature_map3_0[j], conv3[j])+conv3_b[j]  # 第三层输入与第三层卷积核相关，添加偏置
            feature_map3_relu = relu(feature_map3)  # 激活函数
            '''
            池化层P4
            pool:池化函数，图片大小减半添加偏置
            '''
            feature_map4 = pool(feature_map3_relu, 2,
                                pool4_b)  # 池化操作 图片大小减半添加偏置
            feature_map4_relu = relu(feature_map4)  # 激活函数
            '''
            卷积层C5
            '''
            feature_map5_0 = np.zeros((5, 5))  # 存储第五层特征图的输入
            for j in range(16):
                feature_map5_0 = feature_map4_relu[j] + feature_map5_0
            # 卷积层C5输入平均
            # feature_map5_0 = feature_map5_0 / 16
            feature_map5 = np.zeros((120, 1, 1))  # 第五特征图
            for j in range(0, 120):
                feature_map5[j] = convolution(
                    feature_map5_0, conv5[j]) + conv5_b[j][0]  # 第五层特征图输入与第五层卷积核相关，添加偏置
            feature_map5_relu = relu(feature_map5)  # 激活函数
            '''
            全连接层
            '''
            full_input = np.zeros((120, 1))  # 全连接层输入
            for j in range(0, 120):
                # 第五层三维特征图转化为全连接层二维输入
                full_input[j][0] = feature_map5_relu[j][0][0]
            # 全连接层的隐藏层结点 hide_w:84*120 full_input:120*1 hide_b:84*1
            hide = np.dot(hide_w, full_input) + hide_b
            hide_relu = relu_2d(hide)  # 激活函数
            # 全连接层输出节点 output_w:10*84 hide_relu:84*1 output_b:10*1
            output = np.dot(output_w, hide_relu) + output_b
            # print(output)
            '''
            softmax分类
            result 真实结果
            '''
            softmax_result = softmax(
                output)  # 预测结果 softmax_result:10*1 预测各输出可能正确的概率
            # print(softmax_result)
            result = np.zeros((10, 1))  # 存储正确结果
            # print(train_labels[i])
            result[int(train_labels[i])] = 1   # 将result数组中的期望结果置为1
            # lnP （P：softmax_result） 求cross_entropy 预防softmax上溢出和下溢出
            ln_softmax = np.zeros((10, 1))
            output_max = np.max(output)  # 预防softmax上溢出和下溢出算法 临时变量
            denominator = 0  # softmax分母
            for j in range(10):
                denominator = denominator + \
                    np.exp(output[j] - output_max)   # softmax分母计算
            for j in range(10):
                ln_softmax[j] = output[j] - output_max - \
                    np.log(denominator)  # 计算lnP （P：softmax_result）
            cross_entropy = (-1) * \
                np.sum(np.multiply(result, ln_softmax))  # 交叉熵误差公式
            # print(cross_entropy)
            # loss = loss + cross_entropy  # 本块总误差
        # 反向传播
            '''
            全连接隐藏与输出层误差影响
            灵敏度->Loss对该层每个结点的偏导
            sigma_output 灵敏度 10*1
            '''
            # Loss对output层的偏导（总误差对输出结果求导*对激活函数求导）
            sigma_output = softmax_result - result
            # output_b_loss = Loss对output层的偏导 * 1
            output_b_loss = sigma_output + output_b_loss
            # output_w_loss = Loss对output层的偏导 * hide
            # output_w_loss = np.dot(sigma_output, hide.T) + output_w_loss
            for j in range(10):
                for k in range(84):
                    output_w_loss[j][k] = sigma_output[j] * hide[k] + output_w_loss[j][k]
            # np.hide(sum) -> output层输入
            '''
            全连接输入与隐藏层误差影响
            sigma_hide 隐藏层灵敏度 84*1
            '''
            sigma_hide = np.multiply(
                np.dot(output_w.T, sigma_output), relu_back_2d(hide_relu))
            # np.dot(output_w.T, sigma_output) hide与output全连接
            # 每个hide结点影响所有output结点，output所有结点反向影响hide每个结点
            hide_b_loss = sigma_hide + hide_b_loss  # hide_b_loss = Loss对hide层的偏导 * 1
            # hide_w_loss = Loss对hide层的偏导 * np.sum(full_input)
            # hide_w_loss = np.dot(sigma_hide, full_input.T) + hide_w_loss
            for j in range(84):
                for k in range(120):
                    hide_w_loss[j][k] = sigma_hide[j] * full_input[k] +hide_w_loss[j][k]
            # np.hide(input) -> hide层输入
            '''
            第五层卷积与全连接层输入层误差影响
            sigma_conv5 120*1*1 第五层卷积灵敏度
            '''
            sigma_conv5 = np.multiply(
                np.dot(hide_w.T, sigma_hide), relu_back_2d(full_input))  # Loss对conv5层的偏导
            sigma_conv5 = sigma_conv5.reshape(
                [120, 1, 1])  # 将Loss对conv5层的偏导还原为三维
            for j in range(120):
                conv5_loss[j] = convolution(
                    feature_map5_0, sigma_conv5[j]) + conv5_loss[j]
                # conv5_loss = Loss对conv5层的偏导 * conv5层的输入（feature_map_5_0)
                conv5_b_loss[j] = sigma_conv5[j] + conv5_b_loss[j]
                # conv5_b_loss = Loss对conv5层的偏导 * 1
            '''
            第五层卷积与上一池化层误差影响
            sigma_pool4 第四池化层灵敏度  16*5*5
            '''
            sigma_pool4_0 = np.zeros((120, 9, 9))  # 存储上一层灵敏度扩充后的数组
            for j in range(120):
                sigma_pool4_0[j] = np.pad(
                    sigma_conv5[j], 4, 'constant')  # 上一层灵敏度扩充
            # print(sigma_pool4_0.shape)  # 120 * 9 * 9
            # sigma_pool4_1 = convolution_3d(sigma_pool4_0, conv5)  # 上一层灵敏度*权值
            sigma_pool4_1 = np.zeros((120, 5, 5))  # 存储上一层灵敏度*权值
            for j in range(120):
                sigma_pool4_1[j] = convolution(
                    sigma_pool4_0[j], np.rot90(conv5[j], 2))  # 计算该层各结点灵敏度*权值
            sigma_pool4_2 = np.zeros((5, 5))  # 存储上一层灵敏度*权值的和
            for j in range(120):
                sigma_pool4_2 = np.add(
                    sigma_pool4_2, sigma_pool4_1[j])  # 上一层灵敏度*权值 求和
            sigma_pool4 = np.zeros((16, 5, 5))  # 第四层灵敏度
            for j in range(16):
                sigma_pool4[j] = np.multiply(
                    sigma_pool4_2, relu_back_2d(feature_map4_relu[j]))
            # 第四池化层灵敏度->各节点对应的偏导的和(sigma_pool4_2）* 该层激活函数的导数
            # 某节点对应的偏导的和即在计算下一层结点时用到了该结点的所有的结点反向传回的偏导的和 即sigma_pool4_2
            for j in range(16):
                pool4_b_loss[j] = np.sum(sigma_pool4[j]) + pool4_b_loss[j]
            '''
            第四层池化层与上一卷积层误差影响
            sigma_conv3 16*10*10
            '''
            conv3_input = feature_map3_0  # 第三卷积层的输入是第二层结果的组合
            kron_add = np.ones((2, 2)) * (1 / 4)  # 平均池化的反向，一个误差分四份
            sigma_conv3_0 = np.kron(sigma_pool4, kron_add)
            # 平均池化的反向，上一层灵敏度分四份，扩充该层灵敏度大小与该层输入大小相同
            sigma_conv3 = np.multiply(
                sigma_conv3_0, relu_back(feature_map3_relu))
            # sigma_conv3 -> 扩充后的灵敏度*激活函数的导数
            for j in range(16):
                conv3_loss[j] = convolution(
                    conv3_input[j], sigma_conv3[j]) + conv3_loss[j]
                # conv3_loss = Loss对conv3各节点的偏导（灵敏度sigma_conv3 上一池化层求过和)* 第三卷积层的输入
                conv3_b_loss[j] = np.sum(conv3_loss[j]) + conv3_b_loss[j]
            '''
            第三卷积层与上一池化层误差影响
            2->3 : 3*6 4*6 4*3 6*1 =6+6+3+1=16
            十六个输入(conv3_input)对应十六个5*5的卷积核
            sigma_pool2 6*14*14
            '''
            sigma_pool2_0 = np.zeros((16, 18, 18))
            for j in range(16):
                sigma_pool2_0[j] = np.pad(sigma_conv3[j], 4, 'constant')
                # 扩充第三层灵敏度使其与第三层灵敏度卷积后大小与第二层特征图大小（灵敏度大小）一致
            sigma_pool2_1 = np.zeros((16, 14, 14))
            for j in range(16):
                sigma_pool2_1[j] = convolution(
                    sigma_pool2_0[j], np.rot90(conv3[j], 2))
                # 计算该层灵敏度的一部分
            sigma_pool2_2 = np.zeros((6, 14, 14))
            # 某节点对应的偏导的和即在计算下一层结点时用到了该结点的所有的结点反向传回的偏导的和 
            sigma_pool2_2[0] = sigma_pool2_1[0]+sigma_pool2_1[4]+sigma_pool2_1[5]+sigma_pool2_1[6]+sigma_pool2_1[9] + \
                sigma_pool2_1[10]+sigma_pool2_1[11] + \
                sigma_pool2_1[12]+sigma_pool2_1[14]+sigma_pool2_1[15]
            sigma_pool2_2[1] = sigma_pool2_1[0]+sigma_pool2_1[1]+sigma_pool2_1[5]+sigma_pool2_1[6]+sigma_pool2_1[7] + \
                sigma_pool2_1[10]+sigma_pool2_1[11] + \
                sigma_pool2_1[12]+sigma_pool2_1[13]+sigma_pool2_1[15]
            sigma_pool2_2[2] = sigma_pool2_1[0] + sigma_pool2_1[1] + sigma_pool2_1[2] + sigma_pool2_1[6] + sigma_pool2_1[7] + \
                sigma_pool2_1[8] + sigma_pool2_1[11] + \
                sigma_pool2_1[13] + sigma_pool2_1[14] + sigma_pool2_1[15]
            sigma_pool2_2[3] = sigma_pool2_1[1]+sigma_pool2_1[2]+sigma_pool2_1[3]+sigma_pool2_1[6] + \
                sigma_pool2_1[7]+sigma_pool2_1[8]+sigma_pool2_1[9] + \
                sigma_pool2_1[12]+sigma_pool2_1[14]+sigma_pool2_1[15]
            sigma_pool2_2[4] = sigma_pool2_1[2]+sigma_pool2_1[3]+sigma_pool2_1[4]+sigma_pool2_1[7]+sigma_pool2_1[8] + \
                sigma_pool2_1[9]+sigma_pool2_1[10] + \
                sigma_pool2_1[12]+sigma_pool2_1[13]+sigma_pool2_1[15]
            sigma_pool2_2[5] = sigma_pool2_1[3] + sigma_pool2_1[4] + sigma_pool2_1[5] + sigma_pool2_1[8] + sigma_pool2_1[9] + \
                sigma_pool2_1[10] + sigma_pool2_1[11] + \
                sigma_pool2_1[13] + sigma_pool2_1[14] + sigma_pool2_1[15]
            sigma_pool2 = np.zeros((6, 14, 14))
            for j in range(6):
                sigma_pool2[j] = np.multiply(
                    sigma_pool2_2[j], relu_back_2d(feature_map2_relu[j]))
                # 第二池化层灵敏度->各节点对应的偏导的和（sigma_pool2_2）* 该层激活函数的导数
            for j in range(6):
                pool2_b_loss[j] = np.sum(sigma_pool2[j]) + pool2_b_loss[j]
            # print(pool2_b_loss.shape)
            '''
            第二池化层与上一卷积层误差影响
            sigma_conv1 6*28*28
            '''
            sigma_conv1_0 = np.zeros((6, 28, 28)) # 从最后一层传到第一层的灵敏度
            for j in range(6):
                sigma_conv1_0[j] = np.kron(sigma_pool2[j], kron_add)
                # 扩充第二层灵敏度大小与第一层特征图大小一致，并缩小为反传误差的1/4
            sigma_conv1 = np.multiply(
                sigma_conv1_0, relu_back(feature_map1_relu))
            # 第一卷积层灵敏度-> 各结点对应的偏导的和（上层灵敏度）扩充还原后的sigma_conv1_0 * 该层激活函数的导数
            for j in range(6):
                conv1_loss[j] = convolution(
                    train_image, sigma_conv1[j]) + conv1_loss[j]
                # conv1_loss -> sigma_conv1 * 该层输入
                conv1_b_loss[j] = np.sum(sigma_conv1[j]) + conv1_b_loss[j]
            loss = loss + cross_entropy  # 本块总误差

        # 总误差
        print('loss', loss)
        # 将loss记录到learn_rate_change中，计算前后两次误差变化程度
        learn_rate_change[exper] = loss
        # 更新全连接输出层与全连接隐藏层误差
        output_b = output_b - output_b_loss / 100 * learn_rate
        output_w = output_w - output_w_loss / 100 * learn_rate
        # 更新全连接隐藏层与全连接输入层误差
        hide_w = hide_w - hide_w_loss / 100 * learn_rate
        hide_b = hide_b - hide_b_loss / 100 * learn_rate
        # 更新第五层卷积核及偏置
        conv5 = conv5 - conv5_loss / 100 * learn_rate
        conv5_b = conv5_b - conv5_b_loss / 100 * learn_rate
        # 更新第四池化层偏置
        pool4_b = pool4_b - pool4_b_loss / 100 * learn_rate
        # 更新第三层卷积核及偏置
        conv3 = conv3 - conv3_loss / 100 * learn_rate
        conv3_b = conv3_b - conv3_b_loss / 100 * learn_rate
        # 更新第二池化层偏置
        pool2_b = pool2_b - pool2_b_loss / 100 * learn_rate
        # 更新第一层池化层偏置
        conv1 = conv1 - conv1_loss / 100 * learn_rate
        conv1_b = conv1_b - conv1_b_loss / 100 * learn_rate
        # 若前后两次loss小于0.001，则学习率降低
        if exper >= 1:
            if learn_rate_change[exper-1] - learn_rate_change[exper] < 0.01:
                learn_rate = learn_rate / 2
            print('learn_rate', learn_rate)
        if learn_rate <= 1e-6:
            break
        exper = exper + 1  # 计算次数
        print('次数', exper)
        print('----------------5-5-1417-He-BGD-even-rot-hide_grad-output_grad---------------------')

    count = 0
    # 测试过程实验
    for i in range(bitchSize):
        test_image = np.pad(
            test_images[i], 2, 'constant')  # 将28*28的图片扩展为32*32 test_image为扩充后的图片
        # print(test_labels[i])
        '''
        卷积层C1
        '''
        feature_map1 = np.zeros((6, 28, 28))  # 第一层卷积
        for k in range(0, 6):
            feature_map1[k] = convolution(
                train_image, conv1[k]) + conv1_b[k]  # 输入图片分别与六个卷积核相关
        feature_map1_relu = relu(feature_map1)  # 激活函数
        '''
        池化层P2
        pool:池化函数，图片大小减半添加偏置
        '''
        feature_map2 = pool(feature_map1_relu, 2,
                            pool2_b)  # 池化操作 图片大小减半添加偏置
        feature_map2_relu = relu(feature_map2)  # 激活函数
        '''
        卷积层C3
        十六个组合与十六个卷积核一对一相关
        '''
        feature_map3_0 = np.zeros((16, 14, 14))  # 第三层特征图的十六种组合的输入
        feature_map3_0[0] = feature_map2_relu[0] + \
            feature_map2_relu[1] + feature_map2_relu[2]
        feature_map3_0[1] = feature_map2_relu[1] + \
            feature_map2_relu[2] + feature_map2_relu[3]
        feature_map3_0[2] = feature_map2_relu[2] + \
            feature_map2_relu[3] + feature_map2_relu[4]
        feature_map3_0[3] = feature_map2_relu[3] + \
            feature_map2_relu[4] + feature_map2_relu[5]
        feature_map3_0[4] = feature_map2_relu[0] + \
            feature_map2_relu[4] + feature_map2_relu[5]
        feature_map3_0[5] = feature_map2_relu[0] + \
            feature_map2_relu[1] + feature_map2_relu[5]
        feature_map3_0[6] = feature_map2_relu[0] + feature_map2_relu[1] + \
            feature_map2_relu[2] + feature_map2_relu[3]
        feature_map3_0[7] = feature_map2_relu[1] + feature_map2_relu[2] + \
            feature_map2_relu[3] + feature_map2_relu[4]
        feature_map3_0[8] = feature_map2_relu[2] + feature_map2_relu[3] + \
            feature_map2_relu[4] + feature_map2_relu[5]
        feature_map3_0[9] = feature_map2_relu[0] + feature_map2_relu[3] + \
            feature_map2_relu[4] + feature_map2_relu[5]
        feature_map3_0[10] = feature_map2_relu[0] + \
            feature_map2_relu[1] + \
            feature_map2_relu[4] + feature_map2_relu[5]
        feature_map3_0[11] = feature_map2_relu[0] + \
            feature_map2_relu[1] + \
            feature_map2_relu[2] + feature_map2_relu[5]
        feature_map3_0[12] = feature_map2_relu[0] + \
            feature_map2_relu[1] + \
            feature_map2_relu[3] + feature_map2_relu[4]
        feature_map3_0[13] = feature_map2_relu[1] + \
            feature_map2_relu[2] + \
            feature_map2_relu[4] + feature_map2_relu[5]
        feature_map3_0[14] = feature_map2_relu[0] + \
            feature_map2_relu[2] + \
            feature_map2_relu[3] + feature_map2_relu[5]
        feature_map3_0[15] = feature_map2_relu[0] + feature_map2_relu[1] + \
            feature_map2_relu[2] + feature_map2_relu[3] + \
            feature_map2_relu[4] + feature_map2_relu[5]
        # feature_map3_0 = feature_map3_0 / 6  # 输入特征图求和平均
        feature_map3 = np.zeros((16, 10, 10))  # 第三层特征图
        for j in range(16):
            feature_map3[j] = convolution(
                feature_map3_0[j], conv3[j])+conv3_b[j]  # 第三层输入与第三层卷积核相关，添加偏置
        feature_map3_relu = relu(feature_map3)  # 激活函数
        '''
        池化层P4
        pool:池化函数，图片大小减半添加偏置
        '''
        feature_map4 = pool(feature_map3_relu, 2,
                            pool4_b)  # 池化操作 图片大小减半添加偏置
        feature_map4_relu = relu(feature_map4)  # 激活函数
        '''
        卷积层C5
        '''
        feature_map5_0 = np.zeros((5, 5))  # 第五层特征图的输入
        for j in range(16):
            feature_map5_0 = feature_map4_relu[j] + feature_map5_0
        # feature_map5_0 = feature_map5_0 / 16
        feature_map5 = np.zeros((120, 1, 1))  # 第五特征图
        for j in range(0, 120):
            feature_map5[j] = convolution(
                feature_map5_0, conv5[j]) + conv5_b[j][0]  # 第五层特征图输入与第五层卷积核相关，添加偏置
        feature_map5_relu = relu(feature_map5)  # 激活函数
        '''
        全连接层
        '''
        full_input = np.zeros((120, 1))  # 全连接层输入
        for j in range(0, 120):
            # 第五层三维特征图转化为全连接层二维输入
            full_input[j][0] = feature_map5_relu[j][0][0]
        # 全连接层的隐藏层结点 hide_w:84*120 full_input:120*1 hide_b:84*1
        hide = np.dot(hide_w, full_input) + hide_b
        hide_relu = relu_2d(hide)  # 激活函数
        # 全连接层输出节点 output_w:10*84 hide_relu:84*1 output_b:10*1
        output = np.dot(output_w, hide_relu) + output_b
        # print(output)
        '''
         softmax分类
         '''
        softmax_result = softmax(output)  # 预测结果
        # print(softmax_result)
        softmax_result_max = np.max(softmax_result)
        for j in range(10):
            if softmax_result[j] == softmax_result_max:
                ans = j
        print('学习结果：', ans, '正确结果:', test_labels[i])
        if ans == test_labels[i]:
            count = count + 1
    print('正确率：', count / bitchSize * 100, '%')
    # if np.argmax(softmax_result[i])=np.argmax(test_label[i]):
    #   count=count+1
    # print (np.argmax(test_label[i]))
    end = time.time()
    print('运行时间', end - start)


if __name__ == "__main__":
    lenet5()
