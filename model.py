import math
import random
from typing import List, Tuple

from tqdm import tqdm
from sklearn.datasets import load_boston


def train_test_split(x: list, y: list, percent_split=30):
    assert (len(x) == len(y))
    len_data = len(x)
    len_train = int(len_data * ((100 - 30) / 100))
    x_train, x_test = x[:len_train], x[len_train:]
    y_train, y_test = y[:len_train], y[len_train:]

    return x_train, x_test, y_train, y_test


def relu(x: float):
    return max(0.0, x)


def deriv_relu(x):
    if x > 0:
        return 1
    elif x < 0:
        return 0


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))


def feed_forward(data_x: list, model: List[list], func_act='relu'):
    if func_act == 'relu':
        activation = relu
    elif func_act == 'sigmoid':
        activation = sigmoid
    else:
        print('Not this func activation')
        return -1

    input_ = [activation(sum([data_x[j] * model[0][0][j][i] + model[0][1][i] for j in range(len(data_x))])) for i in
              range(len(model[0][0][0]))]

    hidden_layer_1 = [activation(sum([input_[j] * model[1][0][j][i] + model[1][1][i] for j in range(len(input_))])) for
                      i
                      in range(len(model[1][0][0]))]

    output = \
        [activation(sum([hidden_layer_1[j] * model[2][0][j][i] + model[2][1][i] for j in range(len(hidden_layer_1))]))
         for i in range(len(model[2][0][0]))][0]

    return output


def back_propagation(data_x: list, model: List[list], y, y_pred):
    d_L_d_ypred = deriv_mse(y, y_pred)

    sum_input = [sum([data_x[j] * model[0][0][j][i] + model[0][1][i] for j in range(len(data_x))]) for i in
                 range(len(model[0][0][0]))]

    input_ = [relu(sum_) for sum_ in sum_input]

    sum_hidden_layer_1 = [sum([input_[j] * model[1][0][j][i] + model[1][1][i] for j in range(len(input_))]) for i in
                          range(len(model[1][0][0]))]

    hidden_layer_1 = [relu(sum_) for sum_ in sum_hidden_layer_1]

    sum_output = [sum([hidden_layer_1[j] * model[2][0][j][i] + model[2][1][i] for j in range(len(hidden_layer_1))])
                  for i in range(len(model[2][0][0]))][0]

    output = relu(sum_output)

    d_ypred_d_ouput_w = [hidden_layer_1[i] * deriv_relu(sum_output) for i in range(len(model[2][0]))]
    d_ypred_d_b_ouput = deriv_relu(sum_output)
    d_ypred_d_ouput_h = [model[2][0][i][0] * deriv_relu(sum_output) for i in range(len(model[2][0]))]

    d_ypred_d_hidden1_w = [input_[i] * deriv_relu(sum_hidden_layer_1[i]) for i in range(len(model[1][0]))]
    d_ypred_d_b_hidden1 = [deriv_relu(sum_hidden_layer_1[i]) for i in range(len(model[1][1]))]







    return output


def create_layer(n_input: int, n_output: int) -> Tuple[List[list], list]:
    """

    :param n_input:
    :param n_output:
    :return:
    """
    weights = [[random.random() for i in range(n_output)] for i in range(n_input)]
    bias = [random.random() for i in range(n_output)]

    return weights, bias


def build_model(n_features: int):
    """

    :param n_features:
    :return:
    """
    weights_input, bias_input = create_layer(n_features, n_features)

    weights_hidden_1, bias_hidden_1 = create_layer(n_features, n_features)

    weights_ouput, bias_ouput = create_layer(n_features, 1)

    input_ = [weights_input, bias_input]
    layer_hidden_1 = [weights_hidden_1, bias_hidden_1]
    output_ = [weights_ouput, bias_ouput]

    model = [input_, layer_hidden_1, output_]

    return model

def rmse(y_pred: float, y: float):
    """

    :param y:
    :param y_pred:
    :return:
    """
    mse = (y_pred - y) ** 2
    rmse = mse ** (1 / 2)
    return rmse


def deriv_mse(y, y_pred):
    return -2 * (y - y_pred)


# def deriv_rmse():


def generate_data(n_features: int = 10, count_examples: int = 10):
    """

    :param n_features:
    :param count_examples:
    :return:
    """
    x = [[i * random.randint(1, 5) for i in range(1, n_features + 1)] for j in range(1, count_examples + 1)]

    vector_weights = [random.random() for i in range(n_features)]
    # bias = [random.random() for i in range(n_features)]

    y = [[sum([vector_weights[i] * x[j][i] for i in range(n_features)]) for j in range(count_examples)]][0]
    # [x[i].append(y[i]) for i in range(count_examples)]
    data = x
    # return data
    return x, y


# n_features = 5
# n_samples = 100
# epoch = 10
#
# # x, y = generate_data(n_features, n_samples)
# data = load_boston()
# x, y = data.data, data.target
# weight, bias = build_model(len(x[0]))
# x_train, x_test, y_train, y_test = train_test_split(x, y)
# # rmse(100, 10)
# a = 0
# train(x_train, y_train, weight, bias, epoch)


n_features, n_samples = 3, 100
epoch = 10

x, y = generate_data(n_features, n_samples)
model = build_model(len(x[0]))

# for i in tqdm(range(epoch)):
#     for x, y in zip(x, y):
#         y_pred = feed_forward(x[0], model)


# y_pred = feed_forward(x[0], model)
back_propagation(x[0], model, 100, 10)
