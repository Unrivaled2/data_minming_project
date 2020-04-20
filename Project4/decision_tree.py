import numpy as np
import math
import treePlotter

"""
2020春学期中国地质大学（武汉）数据挖掘课程项目四代码

作者：李静涛
qq：2079624548
"""

def read_data():
    """
    读取每个样本点数据
    编码
    0： Iris-virginica
    1：Iris-setosa
    2： Iris-versicolor
    """
    data = []
    with open("./iris.txt", 'r') as f:
        line = f.readline()  # 一行行读取数据
        while line:
            # print(line)
            temp = []
            res = line.split(",")
            temp.append(float(res[0]))
            temp.append(float(res[1]))
            temp.append(float(res[2]))
            temp.append(float(res[3]))  # 读取每个样本点的属性值
            """
            读取类别属性并编码
            """
            if (res[4].__contains__("vir")):
                temp.append(0)
            elif (res[4].__contains__("set")):
                temp.append(1)
            elif (res[4].__contains__("ver")):
                temp.append(2)
            data.append(temp)
            line = f.readline()
        f.close()
    print("读取到的数据: " + str(data))
    return data

def choose_feature_info_gain(data):
    """
    以信息增益为标准获取分类特征
    :param data: 数据集
    :return: 选取的分类特征，例如[1,3,[xxx],[xxx]]代表选取了第一个特征分类，特征以3为界限分类,以及分割得到的两个数据空间
    """
    feature_nums = data.shape[1] - 1 #得到特征个数
    sample_num = data.shape[0] #得到样本点个数
    base_entropy = compute_entropy(data) #计算分割前熵值
    best_info_gain = 0.0
    best_feature_index = 0
    best_feature_value = 0
    best_left_data_space = []
    best_right_data_space = []
    for i in range(feature_nums):
        feature_list = []
        for instance in data:
            feature_list.append(instance[i])
        print("debug1")
        feature_list = list(set(feature_list))
        feature_list.sort()
        for j in range(len(feature_list)-1):
            left_data_space,right_data_space = split_data(data,i,feature_list[j])
            left_data_space_num = len(left_data_space)
            right_data_space_num = len(right_data_space)
            if(left_data_space_num != 0):
                left_data_space_entropy = compute_entropy(left_data_space)
            else:
                left_data_space_entropy = 0

            if(right_data_space_num != 0):
                right_data_space_entropy = compute_entropy(right_data_space)
            else:
                right_data_space_entropy = 0
            new_entropy = left_data_space_num*left_data_space_entropy/sample_num + right_data_space_num*right_data_space_entropy/sample_num
            info_gain = base_entropy - new_entropy
            print("第" + str(i+1) + '个属性' + " 值" + str(feature_list[j]) + "进行划分")
            print("信息增益为: " + str(info_gain))
            print("最佳" + str(i) + "--" + str(best_feature_value))
            if(info_gain > best_info_gain):
                best_info_gain = info_gain
                best_feature_index = i
                best_feature_value = feature_list[j]
                best_left_data_space = left_data_space
                best_right_data_space = right_data_space

    return [best_feature_index,best_feature_value,best_left_data_space,best_right_data_space]

def choose_feature_gini_index(data):
    """
    以基尼指数为标准获取分类特征
    :param data: 数据集
    :return: 选取的分类特征，例如[1,3,[xxx],[xxx]]代表选取了第一个特征分类，特征以3为界限分类,以及分割得到的两个数据空间
    """
    feature_nums = data.shape[1] - 1 #得到特征个数
    sample_num = data.shape[0] #得到样本点个数
    best_gini_index = np.inf
    best_feature_index = 0
    best_feature_value = 0
    best_left_data_space = []
    best_right_data_space = []
    for i in range(feature_nums):
        feature_list = []
        for instance in data:
            feature_list.append(instance[i])
        print("debug1")
        feature_list = list(set(feature_list))
        feature_list.sort()
        for j in range(len(feature_list)-1):
            left_data_space,right_data_space = split_data(data,i,feature_list[j])
            left_data_space_num = len(left_data_space)
            right_data_space_num = len(right_data_space)
            if(left_data_space_num != 0):
                left_data_space_gini_index = compute_gini_index(left_data_space)
            else:
                left_data_space_gini_index = 1

            if(right_data_space_num != 0):
                right_data_space_gini_index = compute_gini_index(right_data_space)
            else:
                right_data_space_gini_index = 1
            gini_index = left_data_space_num*left_data_space_gini_index/sample_num + right_data_space_num*right_data_space_gini_index/sample_num
            if(gini_index < best_gini_index):
                best_gini_index = gini_index
                best_feature_index = i
                best_feature_value = feature_list[j]
                best_left_data_space = left_data_space
                best_right_data_space = right_data_space

    return [best_feature_index,best_feature_value,best_left_data_space,best_right_data_space]

def choose_feature_cart_measure(data):
    """
    以卡特分数为标准获取分类特征
    :param data: 数据集
    :return: 选取的分类特征，例如[1,3,[xxx],[xxx]]代表选取了第一个特征分类，特征以3为界限分类,以及分割得到的两个数据空间
    """
    feature_nums = data.shape[1] - 1 #得到特征个数
    sample_num = data.shape[0] #得到样本点个数
    best_cart_measure = 0.0
    best_feature_index = 0
    best_feature_value = 0
    best_left_data_space = []
    best_right_data_space = []
    for i in range(feature_nums):
        feature_list = []
        for instance in data:
            feature_list.append(instance[i])
        print("debug1")
        feature_list = list(set(feature_list))
        feature_list.sort()
        for j in range(len(feature_list)-1):
            left_data_space,right_data_space = split_data(data,i,feature_list[j])
            cart_measure = compute_cart_measure(left_data_space,right_data_space)
            print("debug5")
            print(cart_measure)
            if(cart_measure > best_cart_measure):
                best_cart_measure = cart_measure
                best_feature_index = i
                best_feature_value = feature_list[j]
                best_left_data_space = left_data_space
                best_right_data_space = right_data_space

    return [best_feature_index,best_feature_value,best_left_data_space,best_right_data_space]


def split_data(data,index,feature_value):
    """
    数据空间根据某个特征值进行划分
    :param data: 数据空间
    :param index: 需要分割的特征值索引
    :param feature_value: 特征值分割点大小
    :return: 分割后的数据空间，包括左节点和右节点
    """
    left_data_space = []
    right_data_space = []

    for instance in data:
        if(instance[index] <= feature_value):
            left_data_space.append(instance)
        else:
            right_data_space.append(instance)

    return [left_data_space,right_data_space]


def compute_entropy(data):
    """
    计算信息熵
    :param data:数据空间
    :return: 计算出来的熵值
    """
    # 统计各类别样本数
    print("计算" + str(data) + "的熵值")
    vir_num = 0.0
    set_num = 0.0
    ver_num = 0.0
    data = np.array(data)
    sample_num = data.shape[0]
    for instance in data:
        class_label = instance[4]
        if(class_label == 0):
            vir_num += 1
        elif(class_label == 1):
            set_num += 1
        elif(class_label == 2):
            ver_num += 1

    p_vir = vir_num/sample_num
    p_set = set_num/sample_num
    p_ver = ver_num/sample_num
    item1 = 0
    item2 = 0
    item3 = 0
    if(p_vir == 0):
        item1 = 0
    else:
        item1 = -p_vir*(math.log2(p_vir))
    if (p_set == 0):
        item2 = 0
    else:
        item2 = -p_set * (math.log2(p_set))
    if (p_ver == 0):
        item3 = 0
    else:
        item3 = -p_ver * (math.log2(p_ver))
    entropy = item1 + item2 + item3
    print(entropy)
    return entropy

def compute_gini_index(data):
    """
    计算基尼指数
    :param data:数据空间
    :return: 计算出来的基尼指数
    """
    # 统计各类别样本数
    vir_num = 0.0
    set_num = 0.0
    ver_num = 0.0
    data = np.array(data)
    sample_num = data.shape[0]
    for instance in data:
        class_label = instance[4]
        if (class_label == 0):
            vir_num += 1
        elif (class_label == 1):
            set_num += 1
        elif (class_label == 2):
            ver_num += 1

    p_vir = vir_num / sample_num
    p_set = set_num / sample_num
    p_ver = ver_num / sample_num

    return (1 - (p_vir*p_vir + p_set*p_set + p_ver*p_ver))

def compute_cart_measure(left_data_space,right_data_space):
    """
    计算cart分数
    :param left_data_space:分割后左侧数据空间
    :param right_data_space: 分割后右侧数据空间
    :return: 计算出来的cart分数
    """
    left_data_space = np.array(left_data_space)
    right_data_space = np.array(right_data_space)
    left_data_space_num = left_data_space.shape[0]
    right_data_space_num = right_data_space.shape[0]
    sample_num = left_data_space_num + right_data_space_num
    """
    计算左右数据空间中各个类别的数量
    """
    left_data_space_vir_num = 0.0
    left_data_space_set_num = 0.0
    left_data_space_ver_num = 0.0
    right_data_space_vir_num = 0.0
    right_data_space_set_num = 0.0
    right_data_space_ver_num = 0.0
    for instance in left_data_space:
        class_label = instance[4]
        if (class_label == 0):
            left_data_space_vir_num += 1
        elif (class_label == 1):
            left_data_space_set_num += 1
        elif (class_label == 2):
            left_data_space_ver_num += 1
    for instance in right_data_space:
        class_label = instance[4]
        if (class_label == 0):
            right_data_space_vir_num += 1
        elif (class_label == 1):
            right_data_space_set_num += 1
        elif (class_label == 2):
            right_data_space_ver_num += 1

    """
    计算左右数据空间中各个类别的概率
    """
    if(left_data_space_num == 0):
        left_data_space_num = np.inf
    if(right_data_space_num == 0):
        right_data_space_num = np.inf

    """
    计算划分出来的两个数据空间中各个类别的概率
    """
    left_data_space_vir_pro = left_data_space_ver_num/left_data_space_num
    left_data_space_set_pro = left_data_space_set_num/left_data_space_num
    left_data_space_ver_pro = left_data_space_ver_num/left_data_space_num
    right_data_space_vir_pro = right_data_space_vir_num/right_data_space_num
    right_data_space_set_pro = right_data_space_set_num/right_data_space_num
    right_data_space_ver_pro = right_data_space_ver_num/right_data_space_num

    item1 = left_data_space_num/sample_num
    item2 = right_data_space_num/sample_num
    if(left_data_space_num == np.inf):
        item1 = 0.0
    if(right_data_space_num == np.inf):
        item2 = 0.0
    cart_measure = 2*(item1) * (item2) \
                   * (abs(left_data_space_vir_pro - right_data_space_vir_pro) + abs(left_data_space_set_pro - right_data_space_set_pro)
                      + abs(left_data_space_ver_pro - right_data_space_ver_pro))

    return cart_measure


def run_decision_tree(data,min_points,demand_purity,measure_method):
    """
    运行决策树算法
    :param data:训练数据集
    :param min_points:每个数据空间最小点数
    :param demand_purity: 要求的最低纯度
    :return: 字典形式表示的决策树
    """
    data = np.array(data)
    sample_number = data.shape[0]
    label_names = ["Iris-virginica","Iris-setosa","Iris-versicolor"]

    #统计各类别样本数
    vir_num = 0.0
    set_num = 0.0
    ver_num = 0.0
    for instance in data:
        class_label = instance[4]
        if(class_label == 0):
            vir_num += 1
        elif(class_label == 1):
            set_num += 1
        elif(class_label == 2):
            ver_num += 1

    purity = max(vir_num,set_num,ver_num)/sample_number #计算纯度
    if(purity >= demand_purity) or (sample_number <= min_points):
        #print("debug3")
        #print(np.where(np.array([vir_num,set_num,ver_num]) == max(vir_num,set_num,ver_num))[0][0])
        label_name = label_names[np.where(np.array([vir_num,set_num,ver_num]) == max(vir_num,set_num,ver_num))[0][0]] #得到该叶节点的划分结果
        return "标签："+str(label_name) + "\n纯度：" + str(purity) + "\n大小：" + str(sample_number)


    if(measure_method == "gini_index"):
        feature_index,feature_value,left_data_space,right_data_space = choose_feature_gini_index(data) #选取分割的特征索引和特征值
    elif(measure_method == 'info_gain'):
        feature_index, feature_value, left_data_space, right_data_space = choose_feature_info_gain(data)  # 选取分割的特征索引和特征值
    elif(measure_method == "cart"):
        feature_index, feature_value, left_data_space, right_data_space = choose_feature_cart_measure(data)  # 选取分割的特征索引和特征值


    choosen_label_name = '特征' + str(feature_index+1) + " <= " + str(feature_value)

    decision_tree = {choosen_label_name:{}}  # 初始化本次函数调用得到的决策树

    left_part = run_decision_tree(left_data_space,min_points,demand_purity,measure_method=measure_method)
    right_part = run_decision_tree(right_data_space,min_points,demand_purity,measure_method=measure_method)

    decision_tree[choosen_label_name]["左边"] = left_part
    decision_tree[choosen_label_name]["右边"] = right_part

    return decision_tree

if __name__ == "__main__":
    data_set = read_data() #读取iris样本点数据集

    tree = run_decision_tree(data_set, 5, 0.95, measure_method="info_gain")
    print(tree)
    treePlotter.info_gain_Tree(tree)


    tree = run_decision_tree(data_set,5,0.95,measure_method="gini_index")
    #print(tree)
    treePlotter.gini_index_Tree(tree)

    tree = run_decision_tree(data_set, 5, 0.95, measure_method="cart")
    # print(tree)
    treePlotter.cart_measure_Tree(tree)

