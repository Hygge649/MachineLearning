#
# documents = ["Human machine interface for lab abc computer applications",
#
# "A survey of user opinion of computer system response time",
#
# "The EPS user interface management system",
#
# "System and human system engineering testing of EPS",
#
# "Relation of user perceived response time to error measurement",
#
# "The generation of random binary unordered trees",
#
# "The intersection graph of paths in trees",
#
# "Graph minors IV Widths of trees and well quasi ordering",
#
# "Graph minors A survey"]
#
# exclude = ['am', 'there','here', 'for', 'of', 'user']
#
# new_doc = [' '.join([word for word in line.split() if word not in exclude]) for line in documents]
#
# print(type(documents))
# print(type(exclude))
# print(new_doc)

list = ["System","human","[", "system engineering","?"]
#print(''.join(filter(str.isalpha, (i for i in list))))
new_list=[]
for i in list:
    #print(type(i))
    list =[ ''.join(filter(str.isalpha,i))]
    print(list)
    #print(type(list))
    new_list += list

while '' in new_list:
    new_list.remove('')

print(new_list)

