import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

np.random.seed(17)

# readfile
path = "./iris.data" #150/5


#1.Read the Flare dataset into a list and shuffle it with the random.shuffle method.
#2.Split the dataset as five folds to do cross-fold validation:
def readfile(path):
    data=pd.read_csv(path)
    data_array = np.array(data)
    np.random.shuffle(data_array)   #shuffle, fixed seed
    #data_array.tolist()
    # length = len(data_array)  #devide into 5 groups
    # m = math.ceil(length/5)
    # data = []
    # for i in range(5):
    #     data.append(data_array[i*m:(i+1)*m])

    return data_array

dataSet = readfile(path)


#
# test = dataSet[0] #30
# print(len(test))
# train = np.concatenate((dataSet[1],dataSet[2],dataSet[3],dataSet[4]))  #120
# print(len(train))



#2.Split the dataset as five folds to do cross-fold validation:
kf = KFold(n_splits=5, shuffle=False,random_state=None) # k=5， shuffle， fixed seed
#get_n_splits([X, y, groups])
#split(X[,Y,groups])  #return index
#do a split, just pass the data, not the tag
train = []
test = []
train_index = []
test_index = []
for train_index1, test_index1 in kf.split(dataSet):
    train_index.append(train_index1)
    test_index.append(test_index1)
    train.append(dataSet[train_index1])
    test.append(dataSet[test_index1])

train = np.array(train)
#print(train[1])
print('train:',train[1][:,0:4])  #just feature not label
print('train_index:',train_index[1])
print('test_label',test[1][:,-1])


#3.Separate the training dataset into three groups by their labels.

def train_class(train):
    classes =[[],[],[]]
    #classes = np.array()
    # class0 = []  #setosa
    # class1 = []  #virginica
    # class2 = []  #versicorlor
    for sample in train:
        if sample[4] == 'Iris-setosa':
            classes[0].append(sample.tolist())
        elif sample[4] =='Iris-virginica':
            classes[1].append(sample.tolist())
        else:
            classes[2].append(sample.tolist())

    for i in range(3):
        classes[i] = np.array(classes[i])[:,0:4]
        #classes[i] = np.array(classes[i],dtype = float)
        classes[i] = classes[i].astype('float')
    return classes

classes = train_class(train[1])
print('sample:',classes[1][1])




#4.Estimate the prior class probability
pw = []
for i in range(3):
    pw.append(len(classes[i])/120)
print('prior pro:',pw)  #[0.35833333333333334, 0.3333333333333333, 0.3]

#5.the conditional probability
#5.1 kernal:gaussian
def gaussian_pdf(x, mu, h):
    #prob_kernal = 1/math.sqrt(2*3.14) * np.exp(-((x-mu)/h)**2/2)
    #np.exp 'Float' object has no attribute 'exp', The default dtype is None, so the type is added to dtype when matrix is generated
    #such as:Xmat = numpy.mat(_x, dtype=float)
    prob_kernal = 1/math.sqrt(2*3.14) * pow(2.718,(-((x-mu)/h)**2/2))
    prob = prob_kernal/pow(h,2)
    return prob

#5.2 the conditional probability
def parzen_window_pdf(x, data, h):
    px = [gaussian_pdf(x, mu=mu, h=h) for mu in data]
    p_con = np.mean(np.array(px), axis=0) #conditional pro
    return np.array(p_con)
    #return np.sum(px)


# 6.The example x is assigned to the label with the maximum conditional probability
def label_x(x,h,prob):  # test
    prob = prob
    result = []
    result_lable = []
    result_max = []
    for j in range(len(x)):
        pmax = 0
        result_p = []
        for i in range(3):
            p = np.sum(pw[i] * prob[j][i])
            if p > pmax:
                flag = i
                pmax = p
            result_p.append(pmax)
        result_max.append(pmax)
        result.append(result_p)

        if flag == 0:
            result_lable.append('Iris-setosa')
        elif flag == 1:
            result_lable.append('Iris-virginica')
        else:
            result_lable.append('Iris-versicolor')

    # return result

    return result, result_max, result_lable

# result, result_max, result_lable = label_x(test[1])
# print(result)
# print(result_max)
# print(result_lable)

def acc(test,h):
    x = np.array(test[:,0:4])
    prob = []
    for sample in x:
        prob_x =[]
        for i in range(3):
            prob_x.append(parzen_window_pdf(sample, classes[i][:,0:4], h=h))
        prob.append(prob_x)
    prob = np.array(prob)

    flag = 0
    result, result_max, result_lable = label_x(test,h,prob)
    for i in range(len(result_lable)):
        if result_lable[i] == test[:,-1][i]: #test =test[1]
            flag +=1

    acc = flag / len(result_lable)

    return acc


#7.Show the figure that the accuracy performance changes with the h
x_axis = np.linspace(1,10,200) #h
print(type(x_axis[1]))
y_axis = []
for i in range(200):
    y_axis.append(acc(test[1],x_axis[i]))#accuracy
plt.plot(x_axis, y_axis,color='orangered', marker='o', linestyle='-', label='Parzen window')
plt.legend(loc='upper right')
plt.xlabel('Hyperparameter: h')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('parzen_window')
