import tensorflow as tf
from tensorflow.keras import layers

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


def sigmoid_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='sigmoid'))
    return model


def create_hmcnf_model(features_size, label_size, hierarchy, beta):

    features = layers.Input(shape=(features_size,))
    global_models = []
    local_models = []

    for i in range(len(hierarchy)):
        if i == 0:
            global_models.append(global_model()(features))
        else:
            global_models.append(global_model()(layers.concatenate([global_models[i-1], features])))

    p_glob = sigmoid_model(label_size)(global_models[-1])

    for i in range(len(hierarchy)):
        local_models.append(local_model(hierarchy[i])(global_models[i]))

    p_loc = layers.concatenate(local_models)

    labels = layers.Add()([(1-beta) * p_glob, beta * p_loc])

    model = tf.keras.Model(inputs=[features], outputs=[labels])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['mae'])

    return model

