import numpy as np


class Layer:
    def __init__(self, input_links: int, output_links: int, activation: str = 'sigmoid', optimizer: str = 'gd',
                 learning_rate: float = 0.1):
        self.weight = np.random.normal(0, 1, (input_links + 1, output_links))
        self.momentum = np.zeros((input_links + 1, output_links))
        self.sum_of_squares_of_gradients = np.zeros(
            (input_links + 1, output_links))
        self.moving_average_m = np.zeros((input_links + 1, output_links))
        self.moving_average_v = np.zeros((input_links + 1, output_links))
        self.update_times = 1
        self.forward_gradient = None
        self.backward_gradient = None
        self.output = None
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Forward feed
        # y = wx + b
        self.forward_gradient = np.append(
            inputs, np.ones((inputs.shape[0], 1)), axis=1)
        
        if self.activation == 'sigmoid':
            self.output = self.sigmoid(
                np.matmul(self.forward_gradient, self.weight))
        elif self.activation == 'tanh':
            self.output = self.tanh(
                np.matmul(self.forward_gradient, self.weight))
        elif self.activation == 'relu':
            self.output = self.relu(
                np.matmul(self.forward_gradient, self.weight))
        elif self.activation == 'leaky_relu':
            self.output = self.leaky_relu(
                np.matmul(self.forward_gradient, self.weight))
        else:
            # Without activation function
            self.output = np.matmul(self.forward_gradient, self.weight)

        return self.output

    def backward(self, derivative_loss: np.ndarray) -> np.ndarray:
        # Backward propagation
        if self.activation == 'sigmoid':
            self.backward_gradient = np.multiply(
                self.derivative_sigmoid(self.output), derivative_loss)
        elif self.activation == 'tanh':
            self.backward_gradient = np.multiply(
                self.derivative_tanh(self.output), derivative_loss)
        elif self.activation == 'relu':
            self.backward_gradient = np.multiply(
                self.derivative_relu(self.output), derivative_loss)
        elif self.activation == 'leaky_relu':
            self.backward_gradient = np.multiply(
                self.derivative_leaky_relu(self.output), derivative_loss)
        else:
            # Without activation function
            self.backward_gradient = derivative_loss

        return np.matmul(self.backward_gradient, self.weight[:-1].T)

    def update(self) -> None:
        # Update weights
        gradient = np.matmul(self.forward_gradient.T, self.backward_gradient)
        
        # Optimizer
        if self.optimizer == 'momentum':
            self.momentum = 0.9 * self.momentum - self.learning_rate * gradient
            delta_weight = self.momentum
        elif self.optimizer == 'adagrad':
            self.sum_of_squares_of_gradients += np.square(gradient)
            delta_weight = -self.learning_rate * gradient / \
                np.sqrt(self.sum_of_squares_of_gradients + 1e-8)
        elif self.optimizer == 'adam':
            self.moving_average_m = 0.9 * self.moving_average_m + 0.1 * gradient
            self.moving_average_v = 0.999 * \
                self.moving_average_v + 0.001 * np.square(gradient)
            bias_correction_m = self.moving_average_m / \
                (1.0 - 0.9 ** self.update_times)
            bias_correction_v = self.moving_average_v / \
                (1.0 - 0.999 ** self.update_times)
            self.update_times += 1
            delta_weight = -self.learning_rate * bias_correction_m / \
                (np.sqrt(bias_correction_v) + 1e-8)
        else:
            # Without optimizer
            delta_weight = -self.learning_rate * gradient

        self.weight += delta_weight

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Calculate sigmoid function
        # y = 1 / (1 + e^(-x))
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(y: np.ndarray) -> np.ndarray:
        # Calculate the derivative of sigmoid function
        # y' = y(1 - y)
        return np.multiply(y, 1.0 - y)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        # Calculate tanh function
        # y = tanh(x)
        return np.tanh(x)

    @staticmethod
    def derivative_tanh(y: np.ndarray) -> np.ndarray:
        # Calculate the derivative of tanh function
        # y' = 1 - y^2
        return 1.0 - y ** 2

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        # Calculate relu function
        # y = max(0, x)
        return np.maximum(0.0, x)

    @staticmethod
    def derivative_relu(y: np.ndarray) -> np.ndarray:
        # Calculate the derivative of relu function
        # y' = 1 if y > 0
        # y' = 0 if y <= 0
        return np.heaviside(y, 0.0)

    @staticmethod
    def leaky_relu(x: np.ndarray) -> np.ndarray:
        # Calculate leaky relu function
        # y = max(0, x) + 0.01 * min(0, x)
        return np.maximum(0.0, x) + 0.01 * np.minimum(0.0, x)

    @staticmethod
    def derivative_leaky_relu(y: np.ndarray) -> np.ndarray:
        # Calculate the derivative of leaky relu function
        # y' = 1 if y > 0
        # y' = 0.01 if y <= 0
        y[y > 0.0] = 1.0
        y[y <= 0.0] = 0.01
        return y
