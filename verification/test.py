import matlab.engine
import numpy as np
import Utils as util
from NeuralNetwork import NeuralNetwork
from boundprop import bound_prop
from sklearn import datasets
import pickle


def _load_weights(path):
    with util.open_file(path, "rb") as f:
        data = pickle.load(f)
    return data


def _load_weights_pkl(path):
    with util.open_file(path, 'rb') as f:
        nn_f = np.load(f, allow_pickle=True)
        W = []
        b = []
        for layer in nn_f:
            W.append(layer[0].T)
            b.append(np.expand_dims(layer[1], axis=1))
        weights = W
        bias = b
    return weights, bias


def test(eps, dataset, dims, nn_file, input_bounds, num, method, output):
    """
    launch the test, currently support MNIST and iris dataset
    :param eps: epsilon for the attack
    :param dataset: name of the dataset, e.g. MNIST/iris
    :param nn_file: neural network .pkl file
    :param input_bounds: the bounds for normalizing the input
    :param output: output file address
    :return:
    """
    X = 0  # store the training samples
    y = 0  # store the labels
    weights = 0  # store the weights
    bias = 0  # store the bias
    params = 0  # store the params for bound propagation

    if dataset == "MNIST":
        # MNIST dataset
        with util.open_file("model/x_test.npy", 'rb') as f:
            X = np.load(f, allow_pickle=True)
        with util.open_file("model/y_test.npy", 'rb') as f:
            y = np.load(f, allow_pickle=True)
    elif dataset == "iris":
        # Iris dataset
        data_ = datasets.load_iris()
        X = data_['data']
        y = data_['target']
    # extract neural network and store the weights and bias into an instance
    # params = _load_weights(nn_file)
    weights, bias = _load_weights_pkl(nn_file)
    assert len(weights) == len(bias)
    print("neural network: ", nn_file)
    print("dims: ", dims)

    # create a new instance and assign the weights and bias. Although we didn't explicitly use nn.weights/nn.bias here, it should still be done.
    nn = NeuralNetwork(dims)
    nn.weights = weights
    nn.bias = bias

    # start connecting matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r"matlab")
    eng.addpath(r'matlab')
    eng.addpath(r'C:\\Program Files\\Mosek\\10.0\\toolbox\\r2017aom')
    eng.addpath(r'C:\\Program Files\\Mosek\\10.0\\toolbox\\r2017a')
    eng.addpath(r'C:\\Users\\heyia\\convex_packages\\YALMIP-master')
    
    # if you have not installed yalmip, you could just use eng.init() to install
    # eng.init()

    # write data into .mat file
    util.write_single_data_to_matlab_path('./matlab/weights.mat', "weights", weights)
    util.write_single_data_to_matlab_path('./matlab/ias.mat', 'bias', bias)

    solved_primal = 0
    solved_dual = 0
    solved_plus = 0

    for index, _ in enumerate(X):
        original_sample_image = X[index]
        sample_image = np.reshape(original_sample_image, (dims[0], 1))
        util.write_single_data_to_matlab_path('./matlab/sample.mat', 'input', sample_image)
        dims_double = matlab.double(dims)
        sample_label = int(y[index])

        # generate a random target
        np.random.seed(index)
        target = np.random.randint(0, dims[-1])
        if target == sample_label:
            target = (target + 1) % dims[-1]

        # use crown to get the bounds
        bounds = bound_prop(weights, bias, sample_image, eps, input_bounds)

        # write bounds into the .mat file
        util.write_single_data_to_matlab_path('./matlab/y_min.mat', 'y_min', bounds[0])
        util.write_single_data_to_matlab_path('./matlab/y_max.mat', 'y_max', bounds[1])
        util.write_single_data_to_matlab_path('./matlab/x_min.mat', 'x_min', bounds[2])
        util.write_single_data_to_matlab_path('./matlab/x_max.mat', 'x_max', bounds[3])

        print("No." + str(index) + " sample target label is " + str(target) + " true label is " + str(sample_label))

        # different methods have different returned data
        if method == "primal":
            # SDR
            res_primal = eng.test_mnist(eps, dims_double, sample_label + 1, target + 1, input_bounds[0],
                                        input_bounds[1], num, 1, nargout=3)
            if res_primal[2] == 1.0:
                solved_primal += 1

            ret = {
                "target": target,
                "label": sample_label,
                "model_name": nn_file,
                "Primal": res_primal[0],
                "Primal_time": res_primal[1],
                "status_primal": res_primal[2],
            }
        elif method == "dual":
            # DeepSDP
            res_dual = eng.test_mnist(eps, dims_double, sample_label + 1, target + 1, input_bounds[0],
                                      input_bounds[1], num, 2, nargout=3)
            if res_dual[2] == 1.0:
                solved_dual += 1

            ret = {
                "target": target,
                "label": sample_label,
                "model_name": nn_file,
                "Dual": res_dual[0],
                "Dual_time": res_dual[1],
                "status_dual": res_dual[2],
            }
        elif method == "sdpnet":
            # SDPNET
            res_plus = eng.test_mnist(eps, dims_double, sample_label + 1, target + 1, input_bounds[0],
                                      input_bounds[1], num, 3, nargout=3)

            if res_plus[2] == 1.0:
                solved_plus += 1

            ret = {
                "target": target,
                "label": sample_label,
                "model_name": nn_file,
                "res_plus": res_plus[0],
                "res_plus_time": res_plus[1],
                "status_res_plus": res_plus[2],
            }
        elif method == "all":
            res_primal = eng.test_mnist(eps, dims_double, sample_label + 1, target + 1, input_bounds[0],
                                        input_bounds[1], num, 1, nargout=3)
            res_dual = eng.test_mnist(eps, dims_double, sample_label + 1, target + 1, input_bounds[0],
                                      input_bounds[1], num, 2, nargout=3)
            res_plus = eng.test_mnist(eps, dims_double, sample_label + 1, target + 1, input_bounds[0],
                                      input_bounds[1], num, 3, nargout=3)

            if res_primal[2] == 1.0:
                solved_primal += 1

            if res_dual[2] == 1.0:
                solved_dual += 1

            if res_plus[2] == 1.0:
                solved_plus += 1

            ret = {
                "target": target,
                "label": sample_label,
                "model_name": nn_file,
                "Primal": res_primal[0],
                "Primal_time": res_primal[1],
                "Dual": res_dual[0],
                "Dual_time": res_dual[1],
                "status_primal": res_primal[2],
                "status_dual": res_dual[2],
                "res_plus": res_plus[0],
                "res_plus_time": res_plus[1],
                "status_res_plus": res_plus[2],
            }
        else:
            print("Please specify a method 1) primal 2) dual 3) sdpnet 4) all")
            break

        with open(str(output), "a+") as f:
            f.write(str(ret))
            f.write("\n")

    with open(str(output), "a+") as f:
        f.write("primal solved number: " + str(solved_primal))
        f.write("\n")
        f.write("Dual solved number: " + str(solved_dual))
        f.write("\n")
        f.write("Dual solved number: " + str(solved_plus))
        f.write("\n")
