import os
from nltk.stem import PorterStemmer
from collections import defaultdict
import re
import math
import numpy as np


def File_split(docpath):
    f = open(docpath, 'r', encoding='Latin1')
    data = f.read().lower().strip(" ").split()  # 1.lower form, 2.str.split()
    return data


def ReadAllFile(path_original):
    datas = []
    data = []
    for i in os.listdir(path_original):
        for j in os.listdir(os.path.join(path_original, i)):
            data = File_split(os.path.join(path_original, i, j))
            datas.append(data)

    return datas


path = 'D:/desktop/cw/ML/assign1-feature-generation/dataset'
read_file = ReadAllFile(path)
#print(read_file)  #二维列表,一个文件一个列表，五类文件存在一个列表中

path_stopwords = 'D:/desktop/cw/ML/assign1-feature-generation/stopwords.txt'
stopwords = set(File_split(path_stopwords))
#print(stopwords)

def single_uniform_file(single_file):  # 只考虑一个文件
    new_file = []
    for alphabet in single_file:
        new_file.append(''.join([j for j in alphabet if j.isalpha()]))  # non-alphabet

    final_file = [[word for word in new_file if word not in stopwords]]

    stemmer = PorterStemmer()  # method, 单词的还原
    uniform_file = []
    for alphabet in final_file[0]:
        j = re.sub(r'[^a-z]', '', alphabet.lower()).strip()  # 'ÿ' -->' '
        if j:
            uniform_file.append(stemmer.stem(j))  # delete ' '

    # return new_file,final_file,uniform_file
    return uniform_file

Files = []
for i in read_file:
    file = []
    file = single_uniform_file(i)
    # print(File)
    Files.append(file)


#print('uniform_file:',Files)  #得到一个划分后的二维文本列表

# 计算每个词的TF值
def word_tf(doc):
    # 词频统计
    word = defaultdict(int)
    for i in doc:
        word[i] = +1

    wordtf = {}
    for i in word:
        word[i] = word[i] / sum(word.values())
    return word


#n_k
def word_idf(list_words):
    doc_num = len(list_words) #N
    word_doc = defaultdict(int)

    for doc in list_words:
        doc = list(set(doc))
        for word in doc:
            word_doc[word] += 1

    return word_doc, doc_num


def feature_select(list_words,save_path):
    word_doc, N = word_idf(list_words)
    word_list = list(word_doc.keys())
    matrix = np.zeros((N,len(word_list)))

    for i in range(len(list_words)):
        word_term_frequency = word_tf(list_words[i])
        doc_words = list(word_term_frequency.keys())
        sum_a_ij =0

        for word in doc_words:
            word_index = word_list.index(word)
            word_tf_idf= word_term_frequency[word] * math.log(N /word_doc[word])
            sum_a_ij += word_tf_idf ** 2
            matrix[i,word_index ] = word_tf_idf

        matrix[i,:] = matrix[i, :] / math.sqrt(sum_a_ij)
    np.savez(save_path,X= matrix)

    return matrix


save_path = 'D:/desktop/cw/ML/assign1-feature-generation/matrix'
word = feature_select(Files,save_path)
print(word)
