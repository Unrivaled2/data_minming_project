import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx

"""
2020春学期中国地质大学（武汉）数据挖掘课程项目三代码

作者：李静涛
qq：2079624548
"""

def gauss_kernelize(x, y, h, dimension):
    """
    使用高斯核进行核密度估计
    :param x: 进行密度估计的点
    :param y: 样本空间中的某个点
    :param h: KDE核密度估计的超参数
    :param dimension: 样本点的特征个数
    :return: 计算出来的核密度
    """
    kernel = np.exp(-(np.linalg.norm(x-y)/h)**2./2.)/((2.*np.pi)**(dimension/2))
    return kernel


def find_attractor(x,data,h,eps):
    """
    使用mean-shift方法寻找样本点x对应的密度吸引子
    :param x: 要寻找对应密度吸引子的样本点
    :param data: 所有样本点
    :param h: KDE核密度估计的超参数
    :param eps: 判断密度吸引子最终收敛的阈值
    :return:[x,density,radius] 最终计算出来的密度吸引子的位置，密度和邻域
    """
    x = np.copy(x)

    n = data.shape[0] #样本点个数
    d = data.shape[1] #样本点特征个数

    """
    最终得到的该样本点与对应密度吸引子之间的距离，这里算为最后三次迭代距离和，这里初始化记录距离的变量
    """
    radius_new = 0.0
    radius_old = 0.0
    radius_old2 = 0.0

    last_density = 0.0 #初始化记录上一次迭代密度值的变量

    while True:
        radius_old3 = radius_old2
        radius_old2 = radius_old
        radius_old = radius_new
        x2 = np.copy(x)

        """
        遍历所有样本点，进行mean-shift迭代
        """
        x3 = np.zeros((1, d))  # 初始化一次mean-shift后得到的样本点位置
        weight_sum = 0.0  # 初始化mean-shift公式的分母
        for j in range(n):
            kernel = gauss_kernelize(x2, data[j], h, d)
            kernel = kernel / (h ** d)
            weight_sum = weight_sum + kernel
            x3 = x3 + (kernel * data[j])
        x3 = x3/weight_sum #得到一次mean-shift后的样本点
        density = weight_sum / n #得到对应密度吸引子的密度值

        x = x3
        error = abs(density - last_density) #计算相邻两次迭代的密度差值
        last_density = density

        radius_new = np.linalg.norm(x - x2)
        radius = radius_old3 + radius_old2 + radius_old + radius_new #计算最近四次迭代的距离和作为密度吸引子的邻域

        if error < eps: #判断收敛后，返回值
            return [x,density,radius]


def run_denclue(data,h=0.3, eps=1e-4, min_density=0.19):
    """
    运行denclue算法
    :param data: 需要进行聚类的数据
    :param h: KDE核密度估计的超参数
    :param eps: 判断密度吸引子最终收敛的阈值
    :param min_density: 密度吸引子点必须满足的最小密度值
    :return:[cluster_info,labels]
    cluster_info:包含每一簇所有相关信息的字典结构
    labels：说明了每个样本点所属的类簇
    """
    """
    参数检查，根据一些研究，h的最佳取值为np.std(data) / 5
    """
    if h is None:
        h = np.std(data) / 5

    n_samples = data.shape[0] #得到样本点个数
    n_features = data.shape[1] #得到样本点的特征数

    """
    初始化每个样本点对应的密度吸引子（极大值点），对应的与密度吸引子的距离，对应的密度吸引子密度值，对应的标签（所属的类簇，噪声为-1）
    """
    density_attractors = np.zeros((n_samples, n_features))
    radii = np.zeros((n_samples, 1))
    density = np.zeros((n_samples, 1))
    labels = -np.ones(data.shape[0])
    eps = 1e-6

    """
    遍历所有样本，计算其对应密度吸引子的位置，密度和邻域
    """
    for i in range(n_samples):
        density_attractors[i], density[i], radii[i] = find_attractor(data[i], data,h=h, eps=eps)

    """
    将每一个样本作为节点构建图，节点的属性信息包括密度吸引点位置，邻域和密度
    """
    g_clusters = nx.Graph()
    for j1 in range(n_samples):
        g_clusters.add_node(j1, attr_dict={'attractor': density_attractors[j1], 'radius': radii[j1],
                                           'density': density[j1]})
        #print("debug")
        #print(density_attractors)
        #print(g_clusters.nodes)

    """
    双重遍历图中的节点，判断：
        如果两个节点的密度吸引子在彼此的邻域内，则进行合并（连接一条边）
    """
    for j1 in range(n_samples):
        for j2 in (x for x in range(n_samples) if x != j1):
            if g_clusters.has_edge(j1, j2):
                continue
            diff = np.linalg.norm(
                g_clusters.node[j1]['attr_dict']['attractor'] - g_clusters.node[j2]['attr_dict']['attractor'])
            if diff <= (g_clusters.node[j1]['attr_dict']['radius'] + g_clusters.node[j1]['attr_dict']['radius']):
                g_clusters.add_edge(j1, j2)

    """
    得到所有的联通分量，即得到所有的簇
    """
    clusters = list(nx.connected_component_subgraphs(g_clusters))
    print("debug2")
    print(clusters)

    """
    遍历所有的簇，得到每一簇的相关信息，存放在字典cluster_info中
    """
    cluster_info = {}
    num_clusters = 0
    for clust in clusters:
        print("debug4")
        print(clust.nodes)
        max_instance = max(clust, key=lambda x: clust.node[x]['attr_dict']['density'])
        max_density = clust.node[max_instance]['attr_dict']['density']
        max_centroid = clust.node[max_instance]['attr_dict']['attractor']

        c_size = len(clust.nodes())
        cluster_info[num_clusters] = {'instances': clust.nodes(),
                                      'size': c_size,
                                      'centroid': max_centroid,
                                      'density': max_density}

        print("debug3")
        print(max_density)
        if max_density >= min_density:#如果该簇的最大密度大于等于最小密度值，则更改其包含样本点对应的簇标签
            for i in clust.nodes:
                labels[i] = num_clusters
        num_clusters += 1

        """
        根据最小密度阈值筛选满足条件的簇
        """

        for i in range(len(cluster_info)):
            if(cluster_info[i]["density"] < min_density):
                del cluster_info[i]

    return [cluster_info,labels]


def compute_cluster_purity(cluster_info,y):
    """
    计算每一簇的纯度
    :param clusters:运行DENCLUE算法得到的簇信息
    :param y: list结构，存储每个样本点对应的类别信息
    :return: 每个簇及其对应的纯度
    """
    clusters_purity = {}
    for cluster in cluster_info:
        vir_num = 0.0 #统计Iris-virginica类别出现个数的变量
        set_num = 0.0 #统计Iris-setosa类别出现个数的变量
        ver_num = 0.0 #统计Iris-versicolor类别出现个数的变量
        cluster_size = 0.0 #统计该簇的点数

        #遍历该簇的所有节点，统计出每个类别出现的个数
        nodes = cluster_info[cluster]['instances']
        for node in nodes:
            #print("debug6")
            #print(node)
            label = y[node]
            label = label.strip()
            if(label.__contains__("vir")):
                vir_num += 1
            elif(label.__contains__("set")):
                set_num += 1
            elif(label.__contains__("ver")):
                ver_num += 1
            cluster_size += 1

        """
        计算出每个类别的纯度
        """
        vir_purity = vir_num/cluster_size
        set_purity = set_num/cluster_size
        ver_purity = ver_num/cluster_size
        #print("debug5")
        #print(vir_purity)
        #print(set_purity)
        #print(ver_purity)

        cluster_purity = max(max(vir_purity,set_purity),ver_purity) #得到该簇的纯度

        clusters_purity[cluster] = cluster_purity

    return clusters_purity


def computePointDistance(x1,x2):
    """
    计算两个样本点之间的距离
    :param x1: 样本点1
    :param x2: 样本点2
    :return: 样本点距离
    """
    x = x1-x2
    x = np.array(x)
    return np.linalg.norm(x,2)


def knn(x,X,n):
    """
    使用knn算法进行密度估计
    :param x: 需要进行核密度估计的样本点
    :param X: 所有样本点集合
    :param n: 选出最近邻的n个点
    :return: 估计出来的密度
    """
    """
    遍历所有样本点，计算x与之距离并存储
    """
    distances = []
    sample_number = np.array(X).shape[0] #得到样本点个数
    for point in X:
        distance = computePointDistance(x,point)
        distances.append(distance)

    nearst_points_index = [] #存储最近邻的点

    """
    遍历distances，找到最近邻的n个点
    """
    distances = np.array(distances)
    max_distacne = 0.0 #n个点中距离最远的点
    for i in range(n):
        min_distance = min(distances)
        print("此次迭代最小距离为: " + str(min_distance))
        print(x)
        if(min_distance > max_distacne):
            max_distacne = min_distance
        index = np.where(distances == min_distance)
        print(X[index])
        nearst_points_index.append(index[0])
        for j in index:
            distances[j] = np.inf

    print("最近邻的n个点索引")
    print(nearst_points_index)

    print("n个点中最远距离")
    print(max_distacne)

    sphere_volume = (8.0/15.0)*np.pi*np.pi*np.power(max_distacne,5) #计算与距离最远的点构成超球体的体积
    print("超球体体积" + str(sphere_volume))

    density = float(n)/(float(sample_number)*sphere_volume)
    return density


if __name__ == "__main__":
    """
    读取每个样本点数据
    """
    x = []
    y = []
    with open("./iris.txt", 'r') as f:
        line = f.readline()  # 一行行读取数据
        while line:
            # print(line)
            temp = []
            res = line.split(",")
            temp.append(float(res[0]))
            temp.append(float(res[1]))
            temp.append(float(res[2]))
            temp.append(float(res[3]))
            x.append(temp) # 读取每个样本点的属性值
            y.append((res[4])) # 读取每个样本点的标签
            line = f.readline()
        f.close()

    """
    运行DENCLUE算法
    """
    X = np.array(x)
    cluster_info,labels  =  run_denclue(X,eps=1e-4)
    print(cluster_info)
    print(labels)
    print("一共有" + str(len(cluster_info)) + "个类簇")
    for i in range(len(cluster_info)):
        print("第"+ str(i) + "个类簇有" + str(cluster_info[i]["size"]) + "个点")

    for i in range(len(cluster_info)):
        print("第"+ str(i) + "个类簇密度吸引子为: " + str(cluster_info[i]["centroid"]))
        print("第"+ str(i) + "个类簇包含的点为: " + str(cluster_info[i]["instances"]))
        print("\n\n")

    """
    计算每一簇的纯度
    """
    clusters_purity = compute_cluster_purity(cluster_info,y)
    print(clusters_purity)
    for i in range(len(clusters_purity)):
        print("第"+ str(i) + "个类簇的纯度为：" + str(clusters_purity[i]))

    """
    knn算法进行密度估计
    """
    X[0] = [6.17438563, 2.8893981 , 4.7249496 , 1.57035205]
    density = knn(X[0],X,30)
    print("点" + str(X[0]) + " knn算法概率密度计算结果为" + str(density))