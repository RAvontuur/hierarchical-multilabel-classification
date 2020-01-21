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
    if i is not None:
        r[i] = 1
    return r


x = [feat(None),
     feat(0),
     feat(0, 0), feat(0, 0, 0), feat(0, 0, 1),
     feat(0, 1), feat(0, 1, 2), feat(0, 1, 3),

     feat(None, 0), feat(None, 0, 0), feat(None, 0, 1),
     feat(None, 1), feat(None, 1, 2), feat(None, 1, 3),
     feat(0, None, 0), feat(0, None, 1),
     feat(0, None, 2), feat(0, None, 3),
     feat(None, None, 0), feat(None, None, 1),
     feat(None, None, 2), feat(None, None, 3),

     feat(1),
     feat(1, 2), feat(1, 2, 4), feat(1, 2, 5),
     feat(1, 3), feat(1, 3, 6), feat(1, 3, 7),

     feat(None, 2), feat(None, 2, 4), feat(None, 2, 5),
     feat(None, 3), feat(None, 3, 6), feat(None, 3, 7),
     feat(1, None, 4), feat(1, None, 5),
     feat(1, None, 6), feat(1, None, 7),
     feat(None, None, 4), feat(None, None, 5),
     feat(None, None, 6), feat(None, None, 7),
     ]
y = [lab(None),
     lab(0),
     lab(2), lab(6), lab(7),
     lab(3), lab(8), lab(9),

     lab(None), lab(6), lab(7),
     lab(None), lab(8), lab(9),
     lab(6), lab(7),
     lab(8), lab(9),
     lab(None), lab(None),
     lab(None), lab(None),

     lab(1),
     lab(4), lab(10), lab(11),
     lab(5), lab(12), lab(13),

     lab(None), lab(10), lab(11),
     lab(None), lab(12), lab(13),
     lab(10), lab(11),
     lab(12), lab(13),
     lab(None), lab(None),
     lab(None), lab(None),
     ]

x = np.repeat(np.array(x, dtype=float), 500, axis=0)
y = np.repeat(np.array(y, dtype=int), 500, axis=0)


def run_test(beta, relu_size, epochs=100):
    print("beta = {}".format(beta))
    model = model_hmcnf.create_hmcnf_model(feature_size, label_size, hierarchy, beta=beta, dropout_rate=0, relu_size=relu_size)

    model.fit([x],
              [y],
              epochs=epochs, batch_size=256)

    return validate(model, x, y)


t1 = run_test(0, 7, epochs=100)
t2 = run_test(0.5, 7, epochs=400)
t3 = run_test(1, 7, epochs=100)
print("results tests")
print(np.array([t1, t2, t3], dtype=float) * 100.0/y.shape[0])
# typical answer [100%, 39.5%, 100%]
# losses [7.3798e-06, 0.0299, 6.4956e-06]
# When beta=0.5 the gradients counteract?