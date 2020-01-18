from arff_reader import ArffReader
import numpy as np

file_name1 = 'cellcycle/cellcycle_FUN.train.arff'
file_name2 = 'cellcycle/cellcycle_FUN.train.arff'
file_name3 = 'cellcycle/cellcycle_FUN.train.arff'

x = []
y = []

for row in ArffReader(file_name1):
    x.append(row[0:-1])
    y.append(row[-1])

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)