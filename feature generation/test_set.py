from nltk.stem.porter import *

stemmer = PorterStemmer()
plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
'died', 'agreed', 'owned', 'humbled', 'sized',
'meeting', 'stating', 'siezing', 'itemization',
'sensational', 'traditional', 'reference', 'colonizer',
'plotted']

singles = [stemmer.stem(plural) for plural in plurals]

print(singles)

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()#method

print(stemmer.stem('working'))

print(stemmer.stem('worked'))

输出：

work

work
