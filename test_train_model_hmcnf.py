from arff_reader import ArffReader
import numpy as np
import model_hmcnf

hierarchy = [2, 4, 8]
feature_size = 14
label_size = 14


def validate(model, x, y):
    y_out = model.predict([x], batch_size=32)
    y_predict = np.where(y_out > 0.5, 1, 0)

    predict_ok = np.where(np.sum(y_predict - y, axis=1) == 0, 1, 0)

    print("{} good out of {} samples".format(np.sum(predict_ok), predict_ok.shape[0]))
    return np.sum(predict_ok)


def feat(i, j=None, k=None):
    r = [0] * feature_size

    if i is not None:
        r[i] = 1
    if j is not None:
        r[hierarchy[0] + j] = 1
    if k is not None:
        r[hierarchy[0] + hierarchy[1] + k] = 1

    return r


def lab(i):
    r = [0] * label_size
    r[i] = 1
    return r


x = [feat(0),
     feat(0, 0), feat(0, 0, 0), feat(0, 0, 1),
     feat(0, 1), feat(0, 1, 2), feat(0, 1, 3),
     feat(1),
     feat(1, 2), feat(1, 2, 4), feat(1, 2, 5),
     feat(1, 3), feat(1, 3, 6), feat(1, 3, 7),
     ]
y =  [lab(0),
      lab(2), lab(6), lab(7),
      lab(3), lab(8), lab(9),
      lab(1),
      lab(4), lab(10), lab(11),
      lab(5), lab(12), lab(13)]

x = np.repeat(np.array(x, dtype=float), 500, axis=0)
y = np.repeat(np.array(y, dtype=int), 500, axis=0)

def run_test(beta):
    print("beta = {}" .format(beta))
    model = model_hmcnf.create_hmcnf_model(feature_size, label_size, hierarchy, beta=0, dropout_rate=0, relu_size=7)


    model.fit([x],
              [y],
              epochs=100, batch_size=256)

    return validate(model, x, y)

t1 = run_test(0)
t2 = run_test(0.5)
t3 = run_test(1)
print("results tests")
print([t1, t2, t3])