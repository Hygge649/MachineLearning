import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


np.random.seed(17)


path = "./iris.data" #150/5

def readfile(path):
    data=pd.read_csv(path)
    data_array = np.array(data)
    np.random.shuffle(data_array)   #shuffle, fixed seed
    return data_array

dataSet = readfile(path)


#2.Split the dataset as five folds to do cross-fold validation:
kf = KFold(n_splits=5, shuffle=False,random_state=None) # k=5， shuffle， fixed seed
train = []
test = []
train_index = []
test_index = []
for train_index1, test_index1 in kf.split(dataSet):

    #print("TRAIN:", train_index, "TEST:", test_index)
    train_index.append(train_index1)
    test_index.append(test_index1)
    train.append(dataSet[train_index1])
    test.append(dataSet[test_index1])

train1 = np.array(train[1])
train = train1[:,0:4]
labels = train1[:,-1]

test1 = np.array(test[1])
test = test1[:,0:4]
test_labels = test1[:,-1]
#print('test_labels: ' ,test_labels)


f=0
#1. For each test example, find K  nearest neighbors from n labeled training examples
def knn(x,data,labels,k):
    # 1.Calculate the distance between x and the training data.
    dis = []
    for i in range(len(data)):
        d = np.sqrt(np.sum((x - data[i]) ** 2))
        dis.append(d)
    dis = np.array(dis)
    # 2.Sort by the distance, return index
    idx = np.argsort(dis)
    #print('index of find K  nearest neighbors from n labeled training examples: ',idx[0:k])
    # 3.The k points with the smallest distance.
    # # 4.Determine the frequency of the category of K sample points
    p = []
    #ki =[]
    k1,k2,k3 = 0,0,0
    for i in range(k):
        if labels[idx[i]] =="Iris-setosa":
            k1 +=1
        elif labels[idx[i]] =="Iris-virginica":
            k2 +=1
        else:
            k3 +=1

    ki = [k1,k2,k3]
    #print(ki)
    for i in range(3):
        p.append(ki[i]/k)
   # print(p)
    #p=np.array(p)
    #p= np.argsort(p)
    p_max= 0
    global f

    for i in range(3):
        if ki[i] >p_max:
            p_max = ki[i]
            f = i
    if f == 0:
        return 'Iris-setosa'
    elif f == 1:
        return 'Iris-virginica'
    elif f == 2:
        return 'Iris-versicolor'


#print(test[1],train[1])
def KNN(K):
    result_lable = []
    for i in range(len(test)):
        result_lable.append(knn(test[i], train, labels, K))

    flag = 0
    for i in range(len(test)):
        if result_lable[i] == test_labels[i]:  # test
                flag += 1

    acc = flag / len(test)

    return acc

#5.Show the figure that the accuracy performance changes with the h
x_axis = np.arange(100) #k
y_axis = []
for i in range(100):
    y_axis.append(KNN(x_axis[i]))#accuracy
plt.plot(x_axis, y_axis,color='red', marker='o', linestyle='-', label='KNN')
plt.legend(loc='lower right')
plt.xlabel('Hyperparameter: k')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('KNN')




