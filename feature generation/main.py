import os

from nltk.stem import PorterStemmer #得到一个统一的形式，created"不一定能还原到"create"，但却可以使"create" 和 "created" ，都得到"creat"。
from nltk.stem import WordNetLemmatizer  #正确还原，自己下载语料库

from collections import defaultdict


#对一个文件进行操作，返回频率字典

path_file = 'D:/desktop/cw/ML/assign1-feature-generation/dataset/alt.atheism/49960'
path_stopwords = 'D:/desktop/cw/ML/assign1-feature-generation/stopwords.txt'

def ReadFile(path):
    f = open(path, 'r', encoding='Latin1')
    datas = f.read().lower().strip(" ").split()  #1.lower form, 2.str.split()
    return datas


file = ReadFile(path_file)
stopwords = ReadFile(path_stopwords)


#print(type(file))# list
print('分割后的原文:',file)     #分割后的原文

# print(ReadFile(path_stopwords))


# new_file = ' '.join([word for word in file if word not in stopwords])  #3.delete stopwords
# print(new_file)  #str  删除停止词


new_file = []
new_file.append([word for word in file if word not in stopwords])
print('删除停止词(list):',new_file)
#print(type(new_file))   #list


#("韩老师真聪明".replace("韩老师", ""))
final_file = []
for i in new_file[0]:
    #print(type(i))
    #print(i)
    #final_file = ' '.join([j for j in i if j.isalnum()]) #字母or数字  ONLY last one why?

    final_file.append(''.join([j for j in i if j.isalnum()]))

#print('最终的输出(去掉非字母）：',final_file)


stemmer = PorterStemmer()#method
uniform_file = []
for i in final_file:
    uniform_file.append(stemmer.stem(i) )
print("uniform_file:",uniform_file)


# lemmatizer = WordNetLemmatizer()
# uniform_file = []
# for i in final_file:
#     uniform_file.append(lemmatizer.lemmatize(i) )
# print("uniform_file:",uniform_file)

def feature_select(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for i in list_words:
        doc_frequency[i] += 1  # i-->word(key),  value -->times

    # 计算每个词的TF值
    word_tf = {}  # 存储每个词的tf值 Term Frequency
    for i in doc_frequency:
        word_tf[i ] =doc_frequency[i ] /sum(doc_frequency.values())  # 1.frequency

    # 计算总数 一个文件举例
    N = sum(doc_frequency.values())

    return word_tf

print('频率字典：',feature_select(uniform_file))

