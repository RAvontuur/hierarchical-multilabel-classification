from arff_reader import ArffReader
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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
# (1628, 499)


def local_model(num_labels):
    model = tf.keras.Sequential()
    model.add(layers.Dense(384, activation='relu'))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(num_labels, activation='sigmoid'))
    return model


def global_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(384, activation='relu'))
    model.add(layers.Dropout(0.6))
    return model


def sigmoid_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(499, activation='sigmoid'))
    return model


features = layers.Input(shape=(77,))
# hierarchy-sizes [18, 80, 178, 142, 77, 4]
i1 = 18
i2 = 80
i3 = 178
i4 = 142
i5 = 77
i6 = 4

g1 = global_model()(features)
g2 = global_model()(layers.concatenate([g1, features]))
g3 = global_model()(layers.concatenate([g2, features]))
g4 = global_model()(layers.concatenate([g3, features]))
g5 = global_model()(layers.concatenate([g4, features]))
g6 = global_model()(layers.concatenate([g5, features]))
p_glob = sigmoid_model()(g6)

l1 = local_model(i1)(g1)
l2 = local_model(i2)(g2)
l3 = local_model(i3)(g3)
l4 = local_model(i4)(g4)
l5 = local_model(i5)(g5)
l6 = local_model(i6)(g6)
p_loc = layers.concatenate([l1, l2, l3, l4, l5, l6])

beta = 0.5
labels = layers.Add()([(1-beta) * p_glob, beta * p_loc])

model = tf.keras.Model(inputs=[features], outputs=[labels])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['mae'])

model.summary()

model.fit([x],
          [y],
          epochs=10000, batch_size=256)
model.save("hmcnf.h5")

