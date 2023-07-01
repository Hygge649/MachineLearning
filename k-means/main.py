import pandas as pd
import numpy as np
np.random.seed(17)
import math
import matplotlib.pyplot as plt

path = "./iris.data"

#Read the Flare dataset into a list and shuffle it with the random.shuffle method.
def readfile(path):
    data1=pd.read_csv(path,names=['sepal_lengh_cm','sepal_width_cm','petal_length_cm','petal_width_cm','class_name'])
    data_array = np.array(data1)  #数组
    #data_list =data_array.tolist()  #list

    #print(len(data_array))
    np.random.shuffle(data_array)   #shuffle, fixed seed
    #print(data_array[10])

    return data_array

dataSet = readfile(path)
truth = dataSet[:,4]
dataSet = dataSet[:,0:4]

#print(truth)
#print(dataSet)

def randCent(dataSet,k):

    m,n = dataSet.shape #m=150,n=4
    centroids = np.zeros((k,n)) #3*4
    index1 = []
    for i in range(k): # 执行三次
        index = int(np.random.uniform(0,m)) # 产生0到150的随机数（在数据集中随机挑一个向量做为质心的初值）
        index1.append(index)
        centroids[i,:] = dataSet[index,:] #把对应行的四个维度传给质心的集合
    return centroids,index1

center,index=randCent(dataSet,3)
print('source center:',center)
#print(index)

l = []
for i in range(len(index)):  #如果取到的是同一类呢？
    l.append(truth[i])
#print(l)

def str_list(truth,l):
    for i in range(len(truth)):
        if truth[i] ==l[0]:
            truth[i] = 0
        elif truth[i] ==l[1]:
            truth[i] = 1
        else:
            truth[i] = 2
    return truth  #ground_truth

truth = str_list(truth,l)


# 欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  # 计算欧氏距离

def cos(x,y):
    return np.dot(x, y)/((np.sqrt(np.sum(x**2)*np.sum(y**2))) +float("1e-8"))


#distance =distEclud(dataSet[0],dataSet[0])
#print(distance)


# k均值聚类算法
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行数150
    # 第一列存每个样本属于哪一簇(四个簇)
    # 第二列存每个样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))  # .mat()创建150*2的矩阵
    clusterChange = True
    # 1.初始化质心centroids
    centroids ,index= randCent(dataSet, k)  # 4*4
    while clusterChange:
        # 样本所属簇不再更新时停止迭代
        clusterChange = False
        # 遍历所有的样本（行数150）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有的质心
            # 2.找出最近的质心
            for j in range(k):
                # 计算该样本到4个质心的欧式距离，找到距离最近的那个质心minIndex
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 3.更新该行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 4.更新质心
        for j in range(k):
            # np.nonzero(x)返回值不为零的元素的下标，它的返回值是一个长度为x.ndim(x的轴数)的元组
            # 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。
            # 矩阵名.A 代表将 矩阵转化为array数组类型

            # 这里取矩阵clusterAssment所有行的第一列，转为一个array数组，与j（簇类标签值）比较，返回true or false
            # 通过np.nonzero产生一个array，其中是对应簇类所有的点的下标值（x个）
            # 再用这些下标值求出dataSet数据集中的对应行，保存为pointsInCluster（x*4）
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取对应簇类所有的点（x*4）
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 求均值，产生新的质心
            # axis=0，那么输出是1行4列，求的是pointsInCluster每一列的平均值，即axis是几，那就表明哪一维度被压缩成1
    print("cluster complete")
    return centroids, clusterAssment

center,assment = KMeans(dataSet, 3)
print('Final center:',center)

predict = np.array(assment[:,0]).flatten()
#print(predict)  #预测结果的列表

def NMI(A,B):
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    NMI = 2.0*MI/(Hx+Hy)
    return MI,NMI

MI,NMI = NMI(truth,predict)
print('MI,NMI is :',MI,NMI)

def draw(data,center,assment):
    length=len(center)
    fig=plt.figure
    data1=data[np.nonzero(assment[:,0].A == 0)[0]]
    data2=data[np.nonzero(assment[:,0].A == 1)[0]]
    data3=data[np.nonzero(assment[:,0].A == 2)[0]]
    # 选取前两个维度绘制原始数据的散点图
    plt.scatter(data1[:,0],data1[:,1],c="yellow",marker='o',label='label0')
    plt.scatter(data2[:,0],data2[:,1],c="green", marker='*', label='label1')
    plt.scatter(data3[:,0],data3[:,1],c="blue", marker='+', label='label2')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center',xy=(center[i,0],center[i,1]),xytext=\
        (center[i,0]+1,center[i,1]+1),arrowprops=dict(facecolor='red'))
    plt.show()

draw(dataSet,center,assment)



















