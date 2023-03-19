import json
import tensorflow as tf
import numpy as np
import matlab.engine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn import datasets


def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # Normalize the images.
    train_images = (train_images / 255)
    test_images = (test_images / 255)
    # Flatten the images.
    train_images = train_images.reshape((-1, 784))
    test_images = test_images.reshape((-1, 784))
    return train_images, train_labels, test_images, test_labels


class NeuralNetwork:
    checkpoint_path = "training_1/cp.ckpt"
    aditi_path = "D:/WORK_SPACE/verification/Neural Network/Aditi/nips_pgd.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    def __init__(self, dims):
        self.model = None
        self.weights = []
        self.bias = []
        self.dims = dims
        self.weights_ = None

    # Creating a Sequential Model and adding the layers
    def create_model(self):
        
        model = Sequential([
            Dense(self.dims[1], activation='relu', input_shape=(self.dims[0],))
        ])
        if len(self.dims) >= 4:
            for i in self.dims[2: -1]:
                model.add(Dense(i, activation="relu"))
        model.add(Dense(self.dims[-1], activation="softmax"))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def train(self, cache, mode):
        if mode == 1:
            train_images, train_labels, test_images, test_labels = load_data()
        if mode == 2:
            train_images = datasets.load_iris()['data']
            train_labels = datasets.load_iris()['target']
        self.create_model()
        print(train_images.shape)
        if cache:
            self.model.load_weights(self.checkpoint_path)
        else:
            self.model.fit(x=train_images, y=train_labels, epochs=50, callbacks=[self.cp_callback])
            self.model.load_weights(self.checkpoint_path)

    def load_weights(self):
        model_json = "D:/WORK_SPACE/verification/Neural Network/Aditi/nips_pgd.json"
        self.create_model()
        # self.model.load_weights(self.aditi_path)
        reader = tf.train.load_checkpoint(self.aditi_path)
        variable_map = reader.get_variable_to_shape_map()
        checkpoint_variable_names = variable_map.keys()

        with tf.gfile.Open(model_json) as f:
            print(f)
            list_model_var = json.load(f)

        net_layer_types = []
        net_weights = []
        net_biases = []

        # Checking validity of the input and adding to list
        for layer_model_var in list_model_var:
            if layer_model_var['type'] not in {'ff', 'ff_relu'}:
                raise ValueError('Invalid layer type in description')
            if (layer_model_var['weight_var'] not in checkpoint_variable_names or
                    layer_model_var['bias_var'] not in checkpoint_variable_names):
                print(layer_model_var['weight_var'])
                print(layer_model_var['bias_var'])
                raise ValueError('Variable names not found in checkpoint')

            layer_weight = reader.get_tensor(layer_model_var['weight_var'])
            layer_bias = reader.get_tensor(layer_model_var['bias_var'])

            if layer_model_var['type'] in {'ff', 'ff_relu'}:
                layer_weight = np.transpose(layer_weight)
                net_layer_types.append(layer_model_var['type'])
                net_weights.append(layer_weight)
                net_biases.append(np.expand_dims(layer_bias.T, axis=1))
        reader.get_tensor(layer_model_var['weight_var'])
        self.weights = net_weights
        self.bias = net_biases
        return

    def read_weights(self):
        W = []
        b = []
        for layer in self.model.layers:
            W.append(matlab.double(layer.get_weights()[0].T.tolist()))
            b.append(matlab.double(np.expand_dims(layer.get_weights()[1], axis=1).tolist()))
            self.weights.append(layer.get_weights()[0].T)
            self.bias.append(np.expand_dims(layer.get_weights()[1], axis=1))
        self.weights_ = W
        self.bias_ = b

    def generateRandomWeights(self):
        for index, dim in enumerate(self.dims[: -1]):
            np.random.seed(index)
            self.weights.append(np.random.randn(self.dims[index + 1], dim) * (1 / np.sqrt(self.dims[0])))
            self.bias.append(np.random.randn(self.dims[index + 1], 1) * (1 / np.sqrt(self.dims[0])))
        return

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def softmax(self, x):

        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def predict_manual_mnist(self, x):
        for i, weights_i in enumerate(self.weights[:-1]):
            y = np.matmul(weights_i, x) + self.bias[i]
            x = self.relu(y)
        final = np.matmul(self.weights[-1], x) + self.bias[-1]
        score = self.softmax(final)
        return score, np.argmax(score)

    def predict_manual_taxi(self, x):
        for i, weights_i in enumerate(self.weights[:]):
            y = np.matmul(weights_i, x) + self.bias[i]
            x = self.relu(y)
        return x

    def interval_arithmetic(self, x_min, x_max):
        X_min = []
        X_max = []
        Y_min = []
        Y_max = []
        X_min.append(x_min.T)
        X_max.append(x_max.T)
        for i in range(len(self.dims) - 2):
            Y_min.append((np.matmul(np.maximum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))),
                                    X_min[i].T) + np.matmul(
                np.minimum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))), X_max[i].T) + self.bias[
                              i]).T)
            Y_max.append((np.matmul(np.maximum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))),
                                    X_max[i].T) + np.matmul(
                np.minimum(self.weights[i], np.zeros((self.dims[i + 1], self.dims[i]))), X_min[i].T) + self.bias[
                              i]).T)

            X_min.append(self.relu(Y_min[i]))
            X_max.append(self.relu(Y_max[i]))

        X_min = np.concatenate(X_min[1:], axis=1)
        X_max = np.concatenate(X_max[1:], axis=1)
        Y_min = np.concatenate(Y_min[:], axis=1)
        Y_max = np.concatenate(Y_max[:], axis=1)
        return Y_min, Y_max, X_min, X_max


    def relu(self, x):
        return np.maximum(x, 0)
