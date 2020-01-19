from arff_reader import ArffReader
import numpy as np
from tensorflow import keras

file_name_train = 'cellcycle/cellcycle_FUN.train.arff'
file_name_validate = 'cellcycle/cellcycle_FUN.valid.arff'
file_name_test = 'cellcycle/cellcycle_FUN.test.arff'


def validate(file_name, model_name):
    x = []
    y = []

    for row in ArffReader(file_name):
        x.append(row[0:-1])
        y.append(row[-1])

    x = np.array(x)
    y = np.array(y)

    model = keras.models.load_model(model_name)
    y_out = model.predict([x], batch_size=32)
    y_predict = np.where(y_out > 0.5, 1, 0)

    predict_ok = np.where(np.sum(y_predict - y, axis=1) == 0, 1, 0)

    print("validated {} , {} good out of {} samples using {}".format(file_name, np.sum(predict_ok), predict_ok.shape[0], model_name))
    return np.sum(predict_ok), predict_ok.shape[0]

validate(file_name_train, 'simple.h5')
validate(file_name_validate, 'simple.h5')
validate(file_name_test, 'simple.h5')
