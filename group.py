import math

def lst_trans0(test_list, n):
    """n: split list into `n` groups
    """
    len_list = len(test_list)
    m = math.ceil(len_list / float(n))
    alist = []
    group_m = -1
    for i in range(len_list):
        if i % m == 0:
            group_m += 1
            alist.append([test_list[i]])
        else:
            alist[group_m].append(test_list[i])
    return alist



def lst_trans1(lst, n):
    m = int(math.ceil(len(lst) / float(n)))
    sp_lst = []
    for i in range(n):
        sp_lst.append(lst[i*m:(i+1)*m])
    return sp_lst

def lst_trans2(lst, n):
    if len(lst) % n != 0:
        m = (len(lst) // n) + 1
    else:
        m = len(lst) // n
    sp_lst = []
    for i in range(n):
        sp_lst.append(lst[i*m:(i+1)*m])
    return sp_lst

test = [3,4,5,6,   7,8,9,10,     15,19,55]

print(lst_trans0(test, 3))

print(lst_trans1(test, 3))

print(lst_trans2(test, 3))