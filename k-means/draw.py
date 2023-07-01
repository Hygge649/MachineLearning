import matplotlib.pyplot as plt

from main import initData
path = "D:/desktop/cw/ML/two/assign2-clustering/iris.data"
data= initData(path)



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