import os

import matlab.engine
import Utils as util
from NeuralNetwork import NeuralNetwork


def test(eps, nn_file, output):
    """
    test
    :param eps:
    :param file_path:
    :return:
    """
    dims, weights, bias, x_min, x_max = util.read_nn(nn_file)

    print("model: ", nn_file)
    print("dims: ", dims)

    # create a new instance and assign the weights and bias. Although we didn't explicitly use nn.weights/nn.bias here, it should still be done.
    nn = NeuralNetwork(dims)
    nn.weights = weights
    nn.bias = bias

    # start matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r"matlab")
    eng.addpath(r'matlab')

    # save weights and bias
    util.write_single_data_to_matlab_path('matlab/weights.mat', "weights", weights)
    util.write_single_data_to_matlab_path('matlab/ias.mat', 'bias', bias)

    # counter for solved instance
    solved_primal = 0
    solved_dual = 0
    solved_plus = 0
    # store all the samples
    sample_file_path = []

    # get all the samples
    for root, dirs, files in os.walk("Dataset/AutoTaxi/"):
        sample_file_path = files

    # iterate all the samples
    for i in sample_file_path:
        sample_image = util.read_sample("Dataset/AutoTaxi/" + i)

        # save sample
        util.write_single_data_to_matlab_path('matlab/sample.mat', 'input', sample_image)

        # convert dims to a matlab.double data structure
        dims_double = matlab.double(dims)

        sample_label = 0

        pred = nn.predict_manual_taxi(sample_image)

        # SDR
        res_primal = eng.test_auto_taxi(eps, float(pred[0][0]), dims_double, sample_label + 1, 1, nargout=3)

        # DeepSDP
        res_dual = eng.test_auto_taxi(eps, float(pred[0][0]), dims_double, sample_label + 1, 2, nargout=3)

        # Deeplus
        res_plus = eng.test_auto_taxi(eps, float(pred[0][0]), dims_double, sample_label + 1, 3, nargout=3)

        print("original value: ", pred[0], end="\n\n")

        if res_primal[2] == 1.0:
            solved_primal += 1

        if res_dual[2] == 1.0:
            solved_dual += 1

        if res_plus[2] == 1.0:
            solved_plus += 1

        ret = {
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
            "pred": pred[0],
        }

        with open(str(output), "a+") as f:
            f.write(str(ret))
            f.write("\n")

    with open(str(output), "a+") as f:
        f.write("primal solved number: " + str(solved_primal))
        f.write("\n")
        f.write("Dual solved number: " + str(solved_dual))
        f.write("\n")
        f.write("Plus solved number: " + str(solved_plus))
        f.write("\n")


if __name__ == "__main__":
    file_Auto_Taxi = ["Neural Network/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet", "Neural Network/AutoTaxi/AutoTaxi_64Relus_200Epochs_OneOutput.nnet", "Neural Network/AutoTaxi/AutoTaxi_128Relus_200Epochs_OneOutput.nnet"]
    epss = [0.016, 0.04, 0.08]
    for eps in epss:
        for nn in file_Auto_Taxi:
            test(eps, nn, "log.txt")