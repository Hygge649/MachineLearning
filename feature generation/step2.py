import os

def ReadTxtName(rootdir):  #txt-->list
    lines = []
    with open(rootdir, 'r',encoding = 'Latin1') as file_to_read:
        while True:
            line = file_to_read.readline().lower()   #lower form
            if not line:
                break
            #line = line.strip('\n') #去掉换行符 -->list
            line = line.strip(" ")
            lines.append(line)
    return lines


stopwords = ReadTxtName('D:/desktop/cw/ML/assign1-feature-generation/stopwords.txt')
#f = ReadTxtName('D:/desktop/cw/ML/assign1-feature-generation/dataset/alt.atheism/49960')
# print(type(f))
print(type(stopwords))

path = 'D:/desktop/cw/ML/assign1-feature-generation/dataset'
for i in os.listdir(path):
    for j in os.listdir(os.path.join(path,i)):
        #f=open(os.path.join(path,i,j),encoding = 'Latin1')
        f = ReadTxtName(os.path.join(path,i,j))
        new_f1 = [' '.join([word for word in line.split() if word not in stopwords]) for line in f]  # 1.delete stopwords
        # new_list = []
        # for i in new_f1: #2.delete  all non-alphabet characters
        #     # print(type(i))
        #     list = [''.join(filter(str.isalpha, i))]
        #     print(list)
        #     # print(type(list))
        #     new_list += list
        #
        # while '' in new_list:  #2.1 delete ‘’
        #     new_list.remove('')
        #     print(new_list)

        print(new_f1)



#print("韩老师真聪明".replace("韩老师", ""))