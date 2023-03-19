import test
import sys
import getopt


def main():
    """
    try the following command for verifying pgdnn
    python3 index.py --dataset MNIST --nnfile model/raghunathan18_pgdnn.pkl --eps 0.1 --dims "[784, 200, 100, 10]" --method "sdpnet" --num 30 --input_bounds "(0., 1.)" --output mnist_log.txt

    try the following command for verifying iris neural network (self-trained)
    python3 index.py --dataset iris --nnfile params.pkl --eps 0.075 --dims "[4,5,10,20,30,40,3]" --num 30 --input_bounds "(0., 10.)" --method "sdpnet"   --output iris_log.txt
    """
    opts, args = getopt.getopt(sys.argv[1:], "h:deo",
                               ["dataset=", "nnfile=", "eps=", "dims=", "input_bounds=", "output=", "num=", "method="])
    dataset = ''
    output = ''
    eps = 0
    dims = []
    neural_network = ""
    input_bounds = ()
    num = 0
    method = ""

    for opt, arg in opts:
        if opt == "h":
            print('usage: python3 index.py --dataset <datasetName> --nnfile <neural network file name> --eps <epsilon> --dims "<dimension>" --num <number of neurons for repeated nonlinearity> --input_bounds "<bounds>" --method "<method name>"   --output <output file>')
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-e", "--eps"):
            eps = eval(arg)
        elif opt == "--dims":
            dims = eval(arg)
        elif opt in ("-o", "--output"):
            output = arg
        elif opt == "--nnfile":
            neural_network = arg
        elif opt == "--input_bounds":
            input_bounds = eval(arg)
        elif opt == "--num":
            num = eval(arg)
        elif opt == "--method":
            method = arg

    print(eps)
    test.test(eps, dataset=dataset, dims=dims, nn_file=neural_network, input_bounds=input_bounds, num=num, method = method,
              output=output)
    return


if __name__ == '__main__':
    main()
