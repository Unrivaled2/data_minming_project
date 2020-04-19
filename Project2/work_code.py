#coding:utf-8
import numpy as np
import math

"""
2020春学期中国地质大学（武汉）数据挖掘课程项目一代码

作者：李静涛
qq：2079624548
"""

x = [] #全局变量，存储每个样本的属性值

def read_data():
    """
    读取iris数据集到全局变量x
    :return:
    """
    with open("./iris.txt", 'r') as f:
        line = f.readline()
        while line:
            # print(line)
            res = line.split(",")
            x.append([float(res[0]),float(res[1]),float(res[2]),float(res[3])])
            line = f.readline()
        f.close()

def compute_kernel_value(x1,x2):
    """
    计算两个样本x1和x2的齐次二次核
    :param x1: 样本
    :param x2: 样本
    :return: 计算出来的齐次二次核
    """
    return math.pow(np.sum(x1*x2),2)

def transform_point_to_feature_space(x):
    """
    将输入空间x转换到可以利用齐次二次核计算的特征空间
    :param x:输入空间的一个样本
    :return:对应特征空间中的值
    """
    feature_space_value = [
        math.pow(x[0],2),math.pow(x[1],2),math.pow(x[2],2),math.pow(x[3],2),
        math.sqrt(2)*x[0]*x[1],math.sqrt(2)*x[2]*x[3],math.sqrt(2)*x[0]*x[2],
        math.sqrt(2)*x[0]*x[3],math.sqrt(2)*x[2]*x[1],math.sqrt(2)*x[3]*x[1]
    ]
    return feature_space_value

def transform_points_to_feature_space():
    """
    将iris数据集中的所有样本都转换到可以使用齐次二次核的特征空间
    :return: 特征空间中的所有样本点
    """
    global x
    feature_space_points = []
    x = np.copy(np.array(x))
    n = x.shape[0]
    for i in range(n):
        instance = x[i]
        feature_space_value = transform_point_to_feature_space(instance)
        feature_space_points.append(feature_space_value)
    return feature_space_points

def compute_kernel_matrix():
    """
    使用齐次二次核计算核矩阵
    :return: 计算出来的核矩阵
    """
    global x
    x = np.copy(np.array(x))
    n = x.shape[0]
    kernel_matrix = np.empty([n,n]) #初始化核矩阵
    for i in range(n):
        for j in range(n):
            x1 = x[i]
            x2 = x[j]
            k = compute_kernel_value(x1,x2) #计算样本x1和x2的齐次二次核
            kernel_matrix[i][j] = k
    return kernel_matrix

def center_kernel_matrix(kernel_matrix):
    """
    中心化核矩阵
    :param kernel_matrix: 中心化前的核矩阵
    :return:中心化后的核矩阵
    """
    temp = np.copy(kernel_matrix)
    commen_item = 1.0*np.sum(kernel_matrix)/(150.0*150.0)  #中心化公式的最后一项
    for i in range(150):
        for j in range(150):
            item1 = 0.0
            item2 = 0.0
            for k in range(150):
                item1 = item1 +  1.0 * temp[i][k]/(150.0)
                item2 = item2 + 1.0 * temp[j][k]/(150.0)
            kernel_matrix[i][j]  = kernel_matrix[i][j] - item1 - item2 + commen_item
    return kernel_matrix

def center_and_normalize_feature_space_points(feature_space_points):
    """
    将特征空间中的点中心化同时规范化
    :param feature_space_points:特征空间中的点
    :return: 处理后的点集
    """
    feature_space_points = np.array(feature_space_points)
    mean = np.mean(feature_space_points,axis=0)#计算均值向量
    print("均值" + str(mean))

    centered_feature_space_points = []
    for instance in feature_space_points:
        v = list(map(lambda x: x[0] - x[1], zip(instance, mean))) #中心化特征空间中的点
        centered_feature_space_points.append(v)
    #print("中心化后特征空间中的点......")
    #print(centered_feature_space_points)

    normalized_feature_space_points = []
    for instance in centered_feature_space_points:
        sum = 0
        for e in instance:
            sum += math.pow(e,2)
        v = np.array(instance)/math.sqrt(sum)# 标准化中心化后的特征空间中的点
        normalized_feature_space_points.append(v)

    return normalized_feature_space_points


def normalize_kernel_matrix(kernel_matrix):
    """
    规范化核矩阵
    :param kernel_matrix:规范化前中心化后的核矩阵
    :return: 规范化后的核矩阵
    """
    temp = np.copy(kernel_matrix)
    for i in range(150):
        for j in range(150):
            kernel_matrix[i][j] = kernel_matrix[i][j]/(math.sqrt(temp[i][i]*temp[j][j]))
    return kernel_matrix

def verify(normalized_feature_space_points,normalized_kernel_matrix):
    """
    验证核技巧（中心化标准化后特征空间中的点积结果等于中心化标准化后核矩阵中对应的值）
    :param normalized_feature_space_points: 经中心化并标准化处理后特征空间中的点
    :param normalized_kernel_matrix: 经中心化并标准化处理后的核矩阵
    """
    for i in range(150):
        for j in range(150):
            value_in_feature_space = np.sum(normalized_feature_space_points[i]*normalized_feature_space_points[j])
            value_in_kernel_matrix = normalized_kernel_matrix[i][j]
            print("验证第" + str(i) + "个点和第" + str(j) + "个点  核矩阵中为：" + str(value_in_kernel_matrix) + " 特征空间点积结果为：" + str(value_in_feature_space))

if __name__ == "__main__":
    """
    第一问的结果
    """
    read_data() #读取数据集
    print("读取到的iris数据集......")
    print(x)

    kernel_matrix = compute_kernel_matrix() #计算核矩阵
    print("核矩阵......")
    print(kernel_matrix)

    centered_kernel_matrix = center_kernel_matrix(kernel_matrix) #中心化核矩阵
    print("中心化后的核矩阵......")
    print(centered_kernel_matrix)

    normalized_kernel_matrix = normalize_kernel_matrix(centered_kernel_matrix) #规范化中心化后的核矩阵
    print("中心化标准化后的核矩阵......")
    print(normalized_kernel_matrix)

    """
    第二问的结果
    """
    feature_space_points = transform_points_to_feature_space()#将iris数据集中的所有样本都转换到可以使用齐次二次核的特征空间
    print("特征空间中的样本点......")
    print(feature_space_points)

    normalized_feature_space_points = center_and_normalize_feature_space_points(feature_space_points)#中心化标准化特征空间中的点
    print("中心化标准化后特征空间中的点......")
    print(normalized_feature_space_points)

    """
    第三问的结果
    """
    verify(normalized_feature_space_points,normalized_kernel_matrix) #验证核技巧（中心化标准化后特征空间中的点积结果等于中心化标准化后核矩阵中对应的值）