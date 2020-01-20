import tensorflow as tf
from tensorflow.keras import layers


def local_model(num_labels, dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_labels, activation='sigmoid'))
    return model


def global_model(dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    return model


def sigmoid_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='sigmoid'))
    return model


def create_hmcnf_model(features_size, label_size, hierarchy, beta=0.5, dropout_rate=0.1, relu_size=384):

    features = layers.Input(shape=(features_size,))
    global_models = []
    local_models = []

    for i in range(len(hierarchy)):
        if i == 0:
            global_models.append(global_model(dropout_rate, relu_size)(features))
        else:
            global_models.append(global_model(dropout_rate, relu_size)(layers.concatenate([global_models[i-1], features])))

    p_glob = sigmoid_model(label_size)(global_models[-1])

    for i in range(len(hierarchy)):
        local_models.append(local_model(hierarchy[i], dropout_rate, relu_size)(global_models[i]))

    p_loc = layers.concatenate(local_models)

    labels = layers.Add()([(1-beta) * p_glob, beta * p_loc])

    model = tf.keras.Model(inputs=[features], outputs=[labels])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['mae'])

    return model

