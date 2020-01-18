from arff_reader import ArffReader
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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

# hierarchy-sizes [18, 80, 178, 142, 77, 4, 0, 0], sum=499
# (1628, 77)
# (1628, 499)

def simple_model():
    layer1 = layers.Dense(1000, input_shape=(77,), activation='relu')
    layer2 = layers.Dense(1000, input_shape=(1000,), activation='relu')
    layer3 = layers.Dense(499)
    model = tf.keras.Sequential()
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    return model


features = layers.Input(shape=(77,))
labels = simple_model()(features)


model = tf.keras.Model(inputs=[features], outputs=[labels])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

model.summary()

model.fit([x],
          [y],
          epochs=500, batch_size=256)
model.save("simple.h5")

