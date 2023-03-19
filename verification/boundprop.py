from Utils import *


class Bound:
    def __init__(self, upper_bounds, lower_bounds):
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds


def relu(x):
    return np.maximum(x, 0)


def IBP(bound, weights, bias):
    x_max = [bound.upper_bounds]
    x_min = [bound.lower_bounds]
    y_min = []
    y_max = []
    pre_activation_bounds = []
    activation_bounds = [bound]
    for i in range(len(weights)):
        y_min.append(np.matmul(np.maximum(weights[i], np.zeros_like(weights[i])), x_min[i]) + np.matmul(
            np.minimum(weights[i], np.zeros_like(weights[i])), x_max[i]) + bias[i])
        y_max.append(np.matmul(np.minimum(weights[i], np.zeros_like(weights[i])), x_min[i]) + np.matmul(
            np.maximum(weights[i], np.zeros_like(weights[i])), x_max[i]) + bias[i])
        pre_activation_bounds.append(Bound(y_max[-1], y_min[-1]))
        if i < len(weights) - 1:
            x_max.append(relu(y_max[i]))
            x_min.append(relu(y_min[i]))
            activation_bounds.append(Bound(x_max[-1], x_min[-1]))
    return pre_activation_bounds, activation_bounds


def compute_alpha(upper, lower):
    ret = np.ones(len(upper))
    upper_squeeze = np.squeeze(np.array(upper), axis=1)
    lower_squeeze = np.absolute(np.squeeze(np.array(lower), axis=1))
    ret[np.greater(upper_squeeze, lower_squeeze)] = 1
    ret[np.greater(lower_squeeze, upper_squeeze)] = 0
    return ret


def compute_alpha_l(lower, upper):
    ret = np.zeros_like(lower)
    ret[upper > np.absolute(lower)] = 1
    return ret


def compute_linear_bounds(pre_activation_bounds, dims):
    lower = pre_activation_bounds.lower_bounds
    upper = pre_activation_bounds.upper_bounds

    alpha_u_item = (upper / (upper - lower))
    alpha_u_item[upper < 0] = 0
    alpha_u_item[lower >= 0] = 1
    alpha_u_item = alpha_u_item.T

    beta_u_item = -lower
    beta_u_item[upper < 0] = 0
    beta_u_item[lower >= 0] = 0
    beta_u_item = beta_u_item.T

    alpha_l_item = compute_alpha_l(lower, upper)
    alpha_l_item[upper < 0] = 0
    alpha_l_item[lower >= 0] = 1
    alpha_l_item = alpha_l_item.T

    beta_l_item = np.zeros_like(lower)
    beta_l_item[upper < 0] = 0
    beta_l_item[lower >= 0] = 0
    beta_l_item = beta_l_item.T

    alpha_u = np.matmul(np.ones((dims[0], 1)), alpha_u_item)
    beta_u = np.matmul(np.ones((dims[0], 1)), beta_u_item)
    alpha_l = np.matmul(np.ones((dims[0], 1)), alpha_l_item)
    beta_l = np.matmul(np.ones((dims[0], 1)), beta_l_item)

    return alpha_l, beta_l, alpha_u, beta_u


def bound_prop(weights, bias, x, epsilon, input_bounds):
    Y_min = []
    Y_max = []
    X_min = []
    X_max = []
    upper_bounds = np.minimum(x + epsilon, input_bounds[1])
    lower_bounds = np.maximum(x - epsilon, input_bounds[0])
    initial_bound = Bound(upper_bounds, lower_bounds)
    pre_activation_bounds, activation_bounds = IBP(initial_bound, weights, bias)

    for layer in range(len(weights)):
        upper_, lower_ = bound_prop_specific_layer(weights[:layer + 1], bias[:layer + 1], pre_activation_bounds,
                                                   upper_bounds, lower_bounds)
        Y_min.append(lower_)
        Y_max.append(upper_)
        X_min.append(relu(lower_))
        X_max.append(relu(upper_))
    return Y_min, Y_max, X_min, X_max


def bound_prop_specific_layer(weights, bias, pre_activation_bounds, upper_bounds, lower_bounds):
    # print(pre_activation_bounds[layer].upper_bounds)
    # print(pre_activation_bounds[layer].lower_bounds)

    # global values
    lambdas_ = []
    deltas_ = []
    Lambdas_ = []
    Omegas_ = []
    omegas_ = []
    thetas_ = []

    # initialize
    Lambda = np.eye(weights[-1].shape[0])
    Omega = np.eye(weights[-1].shape[0])
    current_weight = weights[-1]
    temp_dims = (weights[-1].shape[0], current_weight.shape[-1])

    deltas_.append(np.zeros((weights[-1].shape[0], weights[-1].shape[0])))
    thetas_.append(np.zeros((weights[-1].shape[0], weights[-1].shape[0])))
    Lambdas_.append(Lambda)
    Omegas_.append(Omega)

    for i in range(len(weights) - 1, 0, -1):
        # from layer m-1 to 1
        lambda_ = np.zeros(temp_dims)
        delta_ = np.zeros(temp_dims)
        omega_ = np.zeros(temp_dims)
        theta_ = np.zeros(temp_dims)

        alpha_l, beta_l, alpha_u, beta_u = compute_linear_bounds(pre_activation_bounds[i - 1], temp_dims)
        # if k != 0
        lambda_[np.matmul(Lambda, current_weight) < 0] = alpha_l[np.matmul(Lambda, current_weight) < 0]
        lambda_[np.matmul(Lambda, current_weight) >= 0] = alpha_u[np.matmul(Lambda, current_weight) >= 0]
        omega_[np.matmul(Omega, current_weight) < 0] = alpha_u[np.matmul(Omega, current_weight) < 0]
        omega_[np.matmul(Omega, current_weight) >= 0] = alpha_l[np.matmul(Omega, current_weight) >= 0]

        lambdas_.append(lambda_)
        omegas_.append(omega_)

        delta_[np.matmul(Lambda, current_weight) < 0] = beta_l[np.matmul(Lambda, current_weight) < 0]
        delta_[np.matmul(Lambda, current_weight) >= 0] = beta_u[np.matmul(Lambda, current_weight) >= 0]
        theta_[np.matmul(Omega, current_weight) < 0] = beta_u[np.matmul(Omega, current_weight) < 0]
        theta_[np.matmul(Omega, current_weight) >= 0] = beta_l[np.matmul(Omega, current_weight) >= 0]

        deltas_.append(delta_.T)
        thetas_.append(theta_.T)

        # print(np.matmul(Lambda, current_weight).shape)
        # print(lambda_.shape)

        # update
        Lambda = np.matmul(Lambda, current_weight) * lambda_
        Lambdas_.append(Lambda)
        Omega = np.matmul(Omega, current_weight) * omega_
        Omegas_.append(Omega)
        current_weight = weights[i - 1]
        temp_dims = (weights[-1].shape[0], current_weight.shape[-1])
    lambdas_.append(np.ones((weights[-1].shape[0], weights[0].shape[-1])))
    Lambdas_.append(np.matmul(Lambda, current_weight) * lambdas_[-1])
    omegas_.append(np.ones((weights[-1].shape[0], weights[0].shape[-1])))
    Omegas_.append(np.matmul(Omega, current_weight) * omegas_[-1])

    upper_ = 0
    for i in range(0, len(weights)):
        upper_ += np.diag(
            np.matmul(Lambdas_[i], np.matmul(bias[-i - 1], np.ones((1, weights[-1].shape[0]))) + deltas_[i]))

    lower_ = 0
    for i in range(0, len(weights)):
        lower_ += np.diag(
            np.matmul(Omegas_[i], np.matmul(bias[-i - 1], np.ones((1, weights[-1].shape[0]))) + thetas_[i]))

    upper_ += np.squeeze(np.matmul(Lambdas_[-1], upper_bounds), axis=1)
    lower_ += np.squeeze(np.matmul(Omegas_[-1], lower_bounds), axis=1)

    return upper_, lower_
