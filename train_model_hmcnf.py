from arff_reader import ArffReader
import numpy as np
import model_hmcnf

file_name1 = 'cellcycle/cellcycle_FUN.train.arff'

x = []
y = []

for row in ArffReader(file_name1):
    x.append(row[0:-1])
    y.append(row[-1])

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

# hierarchy-sizes [18, 80, 178, 142, 77, 4, 0, 0], sum=499
# (1628, 77)
# (1628, 499


hierarchy = [18, 80, 178, 142, 77, 4]
feature_size = x.shape[1]
label_size = y.shape[1]
beta = 0.5

model = model_hmcnf.create_hmcnf_model(feature_size, label_size, hierarchy, beta)

model.summary()

model.fit([x],
          [y],
          epochs=100, batch_size=256)
model.save("hmcnf.h5")

