import random
import json

from typing import List

random.seed(42)

"""
1) Ваша задача — построить модель[1] , которая может предсказывать координату y.
Вы можете пройти тесты, если прогнозируемые координаты y находятся в пределах погрешности.
Вы получите комплект train, который нужно использовать для сборки модели.
После того, как вы создадите модель, тесты будут вызывать функцию predict и передавать ей x.
Ошибка будет рассчитана с помощью RMSE .
Нельзя использовать библиотеки: sklearn, pandas, tensorflow, numpy, scipy
example_train_set = [(0, 1),
    (2, 2),
    (4, 3),
    (9, 8),
    (3, 5)]
predicted = [dm.predict(point[0]) for point in example_test_set]

Объяснение
[1] Модель интеллектуального анализа данных создается путем применения алгоритма к данным, но это больше, чем алгоритм 
или контейнер метаданных: это набор данных, статистики и шаблоны, которые можно применять к новым данным для создания 
прогнозов и выводов о взаимосвязях.
"""


class NN:
    """
    class neural network
    """

    def __init__(self, ):
        self.model = {}
        self.output = None

    def feedforward(self):
        mat_mul_layer_1 = self.matmult(self.input, self.weights1)
        self.layer1 = [[i for i in map(self.relu, mat_mul_layer_1[j])] for j in range(len(mat_mul_layer_1))]
        mat_mul_output = self.matmult(self.layer1, self.weights2)
        self.output = [[i for i in map(self.relu, mat_mul_output[j])] for j in range(len(mat_mul_output))]

    def backprop(self):
        """
        Reverse aisle to recalculate weights
        :return:
        """
        mean_error = sum(
            [i[0] for i in [[self.rmse(self.output[i][j], self.y[i][j]) for j in range(len(self.y[i]))] for i in
                            range(len(self.y))]]) / len(self.y)

        print(f"RMSE mean Error: {mean_error}")

        d_weights2 = self.matmult(self.transpon(self.layer1), (
            self.matmult([[i[j] * 2 for j in range(len(i))] for i in self.matrix_subtraction(self.y, self.output)],
                         [[self.relu_derivative(self.output[j][i]) for i in range(len(self.output[j]))] for j in
                          range(len(self.output))])))

        delta_error = self.matrix_subtraction(self.y, self.output)

        mupltip_2 = [[i[j] * 2 for j in range(len(i))] for i in delta_error]
        activat_func_1 = self.matmult(mupltip_2,
                                      [[self.relu_derivative(self.output[j][i]) for i in range(len(self.output[j]))] for
                                       j in
                                       range(len(self.output))])

        activat_func_2 = self.matmult(activat_func_1, self.transpon(self.weights2))

        deriv_all = self.matmult(activat_func_2,
                                 [[self.relu_derivative(self.layer1[i][j]) for j in range(len(self.layer1[i]))] for i in
                                  range(len(self.layer1))])
        d_weights1 = self.matmult(self.transpon(self.input), deriv_all)

        self.weights1 = self.matrix_addition(self.weights1,
                                             [[self.lr * d_weights1[i][j] for j in range(len(d_weights1[i]))] for i in
                                              range(len(d_weights1))])
        self.weights2 = self.matrix_addition(self.weights2,
                                             [[self.lr * d_weights2[i][j] for j in range(len(d_weights2[i]))] for i in
                                              range(len(d_weights2))])

    def fit(self, x, y, lr: int = 0.001, epoch: int = 1500):
        """
        model training
        :param x: data for train
        :param y: target
        :param lr: leatning rate for change weights
        :param epoch: count of epoch
        :return:
        """
        self.input = x
        self.y = y
        self.lr = lr

        self.weights1 = [[random.random() for i in range(2)] for i in range(len(self.input[0]))]
        self.weights2 = [[random.random() for i in range(1)] for i in range(2)]

        for i in range(epoch):
            self.feedforward()
            self.backprop()

    def predict(self, x: list):
        """
        predicting on a sample
        :param x:
        :return:
        """
        mat_mul_layer_1 = self.matmult([x], self.weights1)
        layer1 = [[i for i in map(self.relu, mat_mul_layer_1[j])] for j in range(len(mat_mul_layer_1))]
        mat_mul_output = self.matmult(layer1, self.weights2)
        output = [[i for i in map(self.relu, mat_mul_output[j])] for j in range(len(mat_mul_output))]
        return output

    def save(self, path_model):
        """
        model save
        :param path_model:
        :return:
        """
        self.model['weights1'] = self.weights1
        self.model['weights2'] = self.weights2
        model_json = json.dumps(self.model)
        f = open(path_model, "w")
        f.write(model_json)
        f.close()

    def load(self, path_model):
        """
        load model
        :param path_model:
        :return:
        """
        with open(path_model) as json_file:
            data = json.load(json_file)
        json_file.close()
        self.weights1 = data['weights1']
        self.weights2 = data['weights2']

    @staticmethod
    def relu(x: float) -> float:
        """
        func activation
        :param x:
        :return:
        """
        return max(0.0, x)

    @staticmethod
    def relu_derivative(x) -> int:
        """
        derivativa func active
        :param x:
        :return:
        """
        if x > 0:
            return 1
        elif x < 0:
            return 0

    @staticmethod
    def rmse(y_pred: float, y: float) -> float:
        """
        func loss
        :param y:
        :param y_pred:
        :return:
        """
        mse = (y_pred - y) ** 2
        rmse = mse ** (1 / 2)
        return rmse

    def matmult(self, a: List[list], b: List[list]) -> List[list]:
        """
        matrix multiplication of two matrices
        :param a: first matrix
        :param b: second matrix
        :return:
        """
        zip_b = zip(*b)
        zip_b = list(zip_b)
        return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b))
                 for col_b in zip_b] for row_a in a]

    def transpon(self, matrix: List[list]) -> List[list]:
        """
        matrix transposition
        :param matrix:
        :return:
        """
        transpon_matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
        return transpon_matrix

    def matrix_addition(self, A: List[list], B: List[list]) -> List[list]:
        """
        addition of two matrices
        :param A: first matrix
        :param B: second matrix
        :return:
        """
        rowsA = len(A)
        colsA = len(A[0])
        rowsB = len(B)
        colsB = len(B[0])
        if rowsA != rowsB or colsA != colsB:
            raise ArithmeticError('Matrices are NOT the same size.')
        C = self.zeros_matrix(rowsA, colsB)
        for i in range(rowsA):
            for j in range(colsB):
                C[i][j] = A[i][j] + B[i][j]

        return C

    def matrix_subtraction(self, A: List[list], B: List[list]) -> List[list]:
        """
        subtraction of two matrices
        :param A: first matrix
        :param B: second matrix
        :return:
        """
        rowsA = len(A)
        colsA = len(A[0])
        rowsB = len(B)
        colsB = len(B[0])
        if rowsA != rowsB or colsA != colsB:
            raise ArithmeticError('Matrices are NOT the same size.')
        C = self.zeros_matrix(rowsA, colsB)
        for i in range(rowsA):
            for j in range(colsB):
                C[i][j] = A[i][j] - B[i][j]
        return C

    def zeros_matrix(self, rows, cols) -> List[list]:
        """
        creating a zero matrix
        :param rows: count rows
        :param cols: count collumns
        :return:
        """
        M = []
        while len(M) < rows:
            M.append([])
            while len(M[-1]) < cols:
                M[-1].append(0.0)
        return M


def run(epoch: int = 1500):
    x = [[1, 2, 4], [2, 2, 4], [3, 2, 4], [4, 2, 4]]
    y = [[2], [4], [6], [8]]
    nn = NN()
    nn.fit(x, y)
    # nn.save()
    # nn.load()
    pred = [nn.predict(x[i]) for i in range(len(x))]

    print(f"Actual: {y}")
    print(f"Predict: {pred}")


if __name__ == "__main__":
    run()
