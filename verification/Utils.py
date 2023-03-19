import os
import urllib.request
import numpy as np
import ssl
# import cvxpy as cvx
import scipy.io as scio


def open_file(name, *open_args, **open_kwargs):
    local_path = "./" + name
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    return open(local_path, *open_args, **open_kwargs)


def process_bounds(bounds):
    Y_min = []
    Y_max = []
    X_min = []
    X_max = []

    for bound in bounds[1:-1]:
        X_min.append(np.squeeze(bound[0].T, axis=1))
        X_max.append(np.squeeze(bound[1].T, axis=1))
        Y_min.append(np.squeeze(bound[2].T, axis=1))
        Y_max.append(np.squeeze(bound[3].T, axis=1))

    return Y_min, Y_max, X_min, X_max


def write_single_data_to_matlab_path(filename, key, data):
    """
    write data to the matlab path by indicating key and data
    :param filename: the filename
    :param key: the property/key name
    :param data: the data be transferred
    :return:
    """
    # print(tuple(data))
    return scio.savemat(file_name=filename, mdict={key: data})


def read_nn(file_name):
    """
    parse the .nnet file, extract properties such as dims, weights, bias, input_min, input_max.
    :param file_name: file to be read
    :return: dims, weights, bias, input_min, input_max
    """
    with open(file_name, "r") as f:
        line = f.readline()
        while line.startswith("//"):
            line = f.readline()
        layers, dim_in, dim_out, neuronsize = eval(line)
        dims = list(eval(f.readline()))
        f.readline()  # A flag that is no longer used
        input_min = np.array(list(eval(f.readline())))
        input_max = np.array(list(eval(f.readline())))
        f.readline()
        f.readline()
        weights = []
        bias = []
        for i in dims[1:]:
            current_layer = i
            weights_layer = []
            bias_layer = []
            for j in range(current_layer):
                weights_layer.append(list(eval(f.readline())))
            for j in range(current_layer):
                bias_layer.append(list(eval(f.readline())))
            weights_layer = np.array(weights_layer)
            bias_layer = np.array(bias_layer)
            weights.append(weights_layer)
            bias.append(bias_layer)
        return dims, weights, bias, input_min, input_max


def read_sample(filename):
    """
    read data from npy file, resize it to make it as a vector
    :param filename: the file to be read
    :return: input data vector
    """
    data = np.load(filename)
    data = np.reshape(data, (-1, 1))
    return data
