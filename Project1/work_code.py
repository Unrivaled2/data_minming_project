#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
"""
2020春学期中国地质大学（武汉）数据挖掘课程项目一代码

作者：李静涛
qq：2079624548
"""

mpl.rcParams["font.sans-serif"] = ['SimHei']

def read_data():
    """
    读取magic04数据到一个矩阵
    :return: 包含数据的矩阵
    """
    data = []
    with open("./magic04.txt", 'r') as f:
        line = f.readline()
        while line:
            res = line.split(",")[:10]
            res_float = [float(x) for x in res]
            data.append((res_float))
            line = f.readline()
        f.close()
    return data

def compute_mean(data):
    """
    计算给定数据矩阵data的多元均值
    :param data: 数据矩阵，行是样本，列是属性
    :return: 计算出来的多元均值
    """
    mean = np.mean(data, axis=0)
    return mean

def compute_variance(data):
    """
    计算给定数据矩阵data的多元方差
    :param data: 数据矩阵，行是样本，列是属性
    :return: 计算出来的多元方差
    """
    var = np.var(data, axis=0)
    return var

def center_data(data,mean):
    """
    中心化数据
    :param data:数据矩阵，行是样本，列是属性
    :param mean: 每个属性的均值
    :return: 中心化以后的矩阵
    """
    center_data = []
    for instance in data:
        v = list(map(lambda x: x[0] - x[1], zip(instance, mean)))
        center_data.append(v)
    return center_data


def compute_cosine(centered_data,cov):
    """
    计算第一个属性和第二个属性之间的相关系数，即夹角余弦值
    :param data: 中心化后的样本矩阵
    :param cov: 协方差矩阵
    :return: 第一个属性和第二个属性之间的相关系数，即夹角余弦值
    """
    sigma12 = cov[0][1] #第一个属性和第二个属性之间的协方差
    attribute1 = []#第一个属性
    attribute2 = []#第二个属性
    for instance in centered_data:
        attribute1.append(instance[0])
        attribute2.append(instance[1])
    std1 = np.std(attribute1)
    std2 = np.std(attribute2)

    cosine12 = sigma12/(std1*std2)
    return cosine12


def compute_cov_inner_product(centered_data):
    """
    通过数据矩阵中心化后属性的内积计算协方差矩阵
    :param data:中心化后的数据矩阵，行是样本，列是属性
    :return: 计算出来的协方差矩阵
    """
    transposed_data = np.transpose(centered_data)
    temp = np.matmul(transposed_data,centered_data)
    n = transposed_data.shape[1]
    cov = temp/n
    return cov

def compute_cov_outter_product(centered_data):
    """
    通过数据矩阵中心化后点的外积和计算协方差矩阵
    :param data:中心化后的数据矩阵，行是样本，列是属性
    :return: 计算出来的协方差矩阵
    """
    cov = np.zeros([10,10],dtype=np.float64)
    for instance in centered_data:
        cov += np.outer(np.transpose(instance),instance)
    instance_numbers = np.array(centered_data).shape[0] #计算实例个数
    cov = cov/instance_numbers
    return cov

def plot_scatter(data):
    """
    绘制data矩阵关于第一个属性和第二个属性的散点图
    :param data:数据矩阵
    """

    attribute1 = []  # 第一个属性
    attribute2 = []  # 第二个属性
    for instance in data:
        attribute1.append(instance[0])
        attribute2.append(instance[1])

    plt.title("属性1和属性2的2D散点图")
    plt.xlabel("X1:属性1")
    plt.ylabel("X2:属性2")
    plt.plot(attribute1, attribute2, 'ro')
    plt.show()

#正态分布的概率密度函数。mu是均值，sigma是标准差
def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def plot_norm_distribution(data):
    """
    :param data:数据矩阵，行是样本，列是属性
    绘制第一个属性的正态分布曲线
    """
    attribute1 = []  # 第一个属性
    for instance in data:
        attribute1.append(instance[0])

    mean = np.mean(attribute1)
    std1 = np.std(attribute1)

    x = np.arange(-100, 250, 0.1)

    y = normfun(x, mean, std1)
    plt.plot(x, y)

    plt.title('norm distribution of attribute 1 with mean = ' + str(mean)[0:6] + ", std = " + str(std1)[0:6])
    plt.xlabel('x')
    plt.ylabel('Probability')
    # 输出
    plt.show()

if __name__ == "__main__":
    """
    问题1
    """
    data = read_data() #读取数据
    mean = compute_mean(data) #计算多元均值
    print("多元均值是: " + str(mean))

    """
    问题2
    """
    centered_data = center_data(data,mean)#中心化数据矩阵
    #print("数据中心化后的结果是: " + str(centered_data))
    cov = compute_cov_inner_product(centered_data)
    print("利用属性列的内积计算协方差矩阵：" + str(cov))

    """
    问题3
    """
    cov = compute_cov_outter_product(centered_data)
    print("利用样本点的外积计算协方差矩阵：" + str(cov))

    """
    问题4
    """
    cosine12 = compute_cosine(centered_data,cov)#计算属性1和属性2之间的相关系数，即夹角余弦值
    print("属性1和属性2之间的相关系数(夹角余弦值): " + str(cosine12))
    #plot_scatter(data)#绘制未中心化处理时属性1和属性2的散点图
    plot_scatter(centered_data)#绘制中心化处理后属性1和属性2的散点图


    """
    问题5
    """
    plot_norm_distribution(centered_data)#绘制属性1的正态分布曲线图

    """
    问题6
    """
    var = compute_variance(data)  # 计算多元方差
    min_var = np.min(var)
    min_var_attribute = np.argwhere(var == min_var)
    max_var = np.max(var)
    max_var_attribute = np.argwhere(var == max_var)
    print("多元方差是：" + str(var))
    print("方差最大是: " + str(max_var))
    print("取得最大方差的属性是：第" + str(max_var_attribute[0][0] + 1) + "个属性")
    print("方差最小是: " + str(min_var))
    print("取得最小方差的属性是：第" + str(min_var_attribute[0][0] + 1) + "个属性")


    """
    问题7
    """
    for i in range(10):
        cov[i][i] = 1000
    min_cov = np.min(cov)
    min_cov_attributes = np.argwhere(cov == min_cov)
    for i in range(10):
        cov[i][i] = 0
    max_cov = np.max(cov)
    max_cov_attributes = np.argwhere(cov == max_cov)
    print("协方差最大是: " + str(max_cov))
    print("取得最大协方差的属性对是：" + str(max_cov_attributes[0]+1))
    print("协方差最小是: " + str(min_cov))
    print("取得最小协方差的属性对是：" + str(min_cov_attributes[0]+1))
    sys.exit(0)