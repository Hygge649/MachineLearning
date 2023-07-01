import os

path = 'D:/desktop/cw/ML/assign1-feature-generation/dataset'
for i in os.listdir(path):
    for j in os.listdir(os.path.join(path,i)):
        f=open(os.path.join(path,i,j),encoding = 'Latin1')  #使用os.listdir（）索引文章，使用os.path.join（）拼接得到绝对路径，f=open(）打开文件，指定操作类型(r), 编码形式

#f = open('D:/desktop/cw/ML/assign1-feature-generation/dataset/alt.atheism/49960','r',encoding = 'Latin1') #simple test
        for line in f:

            data = line.strip(" ").split()
            #print(type(data))
            print(data)

        f.close()





