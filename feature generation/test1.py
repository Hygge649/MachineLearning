# TFIDF Representation
def word_tf(word_list):  #输入一个文件，统计词频
    #  f_ik: the frequency of word k in document i
    # 出现次数
    doc_frequency = defaultdict(int)
    word_list = tuple(word_list)

    for i in word_list:  # doc中的word  重复？ 按word在每一个文件中的计算频率
        doc_frequency[i] += 1

    # 计算每个词的TF值   term frequency
    word_term_frequency = {}  # dictionary 存储每个词的tf值
    for i in doc_frequency:
        word_term_frequency[i] = doc_frequency[i] / sum(doc_frequency.values())  # values() 方法，把dict转换成一个包含所有value的list

    return word_term_frequency

#print(word_tf(Files))

def word_doc(list_words):
    # document Frequency n_k
    doc_number = len(list_words)  # N
    word_of_doc = {}
    for doc in list_words:  # doc
        file_set = list(set(i))
        for word in file_set:  # word
            if word in word_of_doc:
                word_of_doc[word] += 1  # n_k :the total number of documents that word k occurs in the dataset
            else:
                word_of_doc[word] = 1
    return word_of_doc, doc_number

#print(word_doc(Files))

def feature_selection(list_words, word_of_doc, doc_number, save_path):
    word_list = list(word_of_doc.keys())
    Matrix = np.zeros(doc_number,len(word_list))

    for i in range(len(list_words): #doc
        word_term_frequency = word_tf(list_words[i])
        sum_a_ij = 0
        for word in list_words[i]:
            word_index =
            word_tf_idf = word_term_frequency[word] * math.log(doc_number/word_of_doc[word])
            sum += word_tf_idf ** 2







    return word_list


def feature_selection(list_words, word_of_doc, doc_number, save_path):
    word_list = list(word_of_doc.keys())
    matrix = np.zeros((doc_number, len(word_list)))

    for i in range(len(list_words[0])):  # times of doc
        tf = word_tf(list_words[i])  # dic  key --->  value: word ---> a_ij
        doc_word = list(tf.keys())  #list？
        print(doc_word)
        count = 0
        for word in doc_word:
            word_index = word_list.index(word)
            word_tf_idf = tf[word] * math.log(doc_number / (word_of_doc[word]))  # 计算每个词的IDF值 Inverse Document Frequency
            count += word_tf_idf ** 2
            matrix[i, word_index] = word_tf_idf
        matrix[i, :] = matrix[i, :] / math.sqrt(count)
    np.savez(save_path, A=matrix)

    return word_list
