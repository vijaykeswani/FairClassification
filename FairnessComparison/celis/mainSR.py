import os,sys
import numpy as np
import json
from StatisticalRate import *

def predict(model, x):
    return np.sign(np.dot(x, model))


def load_json(filename):
    f = json.load(open(filename))
    x = np.array(f["x"])
    y = np.array(f["class"])
    sensitive = dict((k, np.array(v)) for (k,v) in f["sensitive"].items())
    return x, y, sensitive

def main(train_file, test_file, output_file, value):
    x_train, y_train, x_control_train = load_json(train_file)
    x_test, y_test, x_control_test = load_json(test_file)

    sensitive_attrs = list(x_control_train.keys())

    tau = float(value)
    print("Tau : ", tau)
    predictions = StatisticalRate().test_given_data(x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs, tau)   

    #predictions = predict(w, x_test).tolist()
    output_file = open(output_file, "w")
    json.dump(predictions, output_file)
    output_file.close()


if __name__ == '__main__':
    main(*sys.argv[1:])
    exit(0)
