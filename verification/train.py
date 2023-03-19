import getopt
import sys
from sklearn import datasets
from NeuralNetwork import NeuralNetwork
import Utils as util
import numpy as np


def train(dataset, dims):
    if dataset == 'iris':
        data_ = datasets.load_iris()
        X = data_['data']
        y = data_['target']
        nn = NeuralNetwork(dims)
        nn.train(cache=False, mode=2)
    elif dataset == "MNIST":
        nn = NeuralNetwork(dims)
        nn.train(cache=False, mode=1)
    else:
        print("Not Support: " + dataset)
        return
    nn.read_weights()

    params = []
    for i in range(len(dims) - 1):
        params.append(tuple([nn.weights[i].T, np.squeeze(nn.bias[i], axis=1)]))
    import pickle
    pickle.dump(params, open("params.pkl", 'wb'))
    print("params.pkl has been created")

    util.write_single_data_to_matlab_path('matlab/weights.mat', "weights", nn.weights)
    util.write_single_data_to_matlab_path('matlab/ias.mat', 'bias', nn.bias)
    print("weights and bias has been created")

    return


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h:deo",
                               ["dataset=", "dims="])
    dataset = ''
    dims = []

    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            dataset = arg
        elif opt == "--dims":
            dims = eval(arg)
    train(dataset, dims)