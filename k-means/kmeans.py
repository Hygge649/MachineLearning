import numpy as np
import random
from read import initData
from math import sqrt

dataSet = initData()


# 为给定数据集构建一个包含K个随机质心centroids的集合
def randCent(dataSet, k):
    #m, n = dataSet.shape  m=150,n=4
    m = len(dataSet)
    centroids = []

    for i in range(k):  # 执行三次
        index = random.randint(0, m)  # 产生0到150的随机数（在数据集中随机挑一个向量做为质心的初值）
        centroids.append(dataSet[index])  # 把对应行的四个维度传给质心的集合
    return centroids


# 欧氏距离计算
def distEclud(x, y):  #two lists
    for i in range(0, 3):
        count = 0
        count += (float(x[i]) - float(y[i]))** 2
        #count = sum(x[i] -y[i])** 2
    d = sqrt(count)  # 计算欧氏距离
    return d

def mean(list1,list2):
    list = []
    for i in range(4):
        a = (float(list1[i])+float(list2[i]))/2
        list.append(a)
    return list


# kmeans 聚类算法
def KMeans(dataSet, k):

    m = len(dataSet) # 行数150
    clusterAssment = np.mat(np.zeros((m, 2)))  # .mat()创建150*2的矩阵  第一列存每个样本属于哪一簇  第二列存每个样本的到簇的中心点的误差
    clusterChange = True

    # 1.初始化质心centroids
    centroids = randCent(dataSet, k)  # 3*4

    while clusterChange:   # 样本所属簇不再更新时停止迭代
        clusterChange = False
        # 遍历所有的样本（行数150）
        for i in range(m-3):
            minDist = 100000.0
            minIndex = -1

            # 2. 遍历所有的质心 找出最近的质心
            for j in range(k):
                # 计算该样本到3个质心的欧式距离，找到距离最近的那个质心minIndex
                distance = distEclud(centroids[j], dataSet[i])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 3.更新该行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True   #一个点加入类中，中心改变，更新质心
                clusterAssment[i, :] = minIndex, minDist ** 2

        # 4.更新质心
        for j in range(k):
            # np.nonzero(x)返回值不为零的元素的下标，它的返回值是一个长度为x.ndim(x的轴数)的元组
            # 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。
            # 矩阵名.A  将矩阵转化为array数组类型

            # 这里取矩阵clusterAssment所有行的第一列，转为一个array数组，与j（簇类标签值）比较，返回true or false
            # 通过np.nonzero产生一个array，其中是对应簇类所有的点的下标值（x个）
            # 再用这些下标值求出dataSet数据集中的对应行，保存为pointsInCluster（x*4）

            index = np.nonzero(clusterAssment[:, 0].A == j) # 获取对应簇类所有的点（x*4）
            for i in range(len(index)):
                centroids[j] = mean([0,0,0,0], dataSet[i])  # 求均值，产生新的质心
            # axis=0，那么输出是1行4列，求的是pointsInCluster每一列的平均值，即axis是几，那就表明哪一维度被压缩成1
    print("cluster complete")
    return centroids, clusterAssment


KMeans(dataSet, 3)


