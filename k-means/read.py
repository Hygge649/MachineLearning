import random
import pandas as pd

path = "D:/desktop/cw/ML/two/assign2-clustering/iris.data"
def initData(path):

    file = open(path)
    lines = file.read().splitlines()
    file.close()

    random.shuffle(lines)
    data_in = []
    for i in range(len(lines)-1):
        data = str(lines[i]).split(',')
        data_in.append(data)

    return data_in
