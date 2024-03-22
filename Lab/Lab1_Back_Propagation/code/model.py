import numpy as np
import matplotlib.pyplot as plt
import layer


class NeuralNetwork:
    def __init__(self, epoch: int = 1000000, learning_rate: float = 0.1, num_of_layers: int = 2, input_units: int = 2,
                 hidden_units: int = 4, activation: str = 'sigmoid', optimizer: str = 'gd'):
        self.num_of_epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.optimizer = optimizer
        self.learning_epoch, self.learning_loss = list(), list()

        # Setup layers
        self.layers = []
        # Input layer
        self.layers.append(layer.Layer(input_units, hidden_units, activation, optimizer, learning_rate))
        # Hidden layers
        for _ in range(num_of_layers - 1):
            self.layers.append(layer.Layer(hidden_units, hidden_units, activation, optimizer, learning_rate))
        # Output layer
        self.layers.append(layer.Layer(hidden_units, 1, 'sigmoid', optimizer, learning_rate))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Forward pass
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, derivative_loss) -> None:
        # Backward pass
        for layer in self.layers[::-1]:
            derivative_loss = layer.backward(derivative_loss)

    def update(self) -> None:
        # Update weights
        for layer in self.layers:
            layer.update()

    def train(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        for epoch in range(self.num_of_epoch):
            # Forward pass
            prediction = self.forward(inputs)
            # Loss
            loss = self.mse_loss(prediction=prediction, ground_truth=labels)
            # Backward pass
            self.backward(self.mse_derivative_loss(
                prediction=prediction, ground_truth=labels))
            # Update weights
            self.update()

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}\tLoss: {loss}')
                self.learning_epoch.append(epoch)
                self.learning_loss.append(loss)

            if loss < 0.001:
                break

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        # Predict the labels of inputs
        prediction = self.forward(inputs=inputs)
        print(prediction)
        return np.round(prediction)

    def test(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        # Predict the labels of inputs
        pred_labels = self.predict(inputs)
        
        # Print labels and predictions
        for i in range(inputs.shape[0]):
            print(
                f"Point: {i}\tLabel: {labels[i]}\tPrediction: {pred_labels[i]}")
        
        # Print Config
        print(f'Accuracy: {float(np.sum(pred_labels == labels)) / len(labels)}')
        print(f'Epoch: {self.num_of_epoch}')
        print(f'Hidden units: {self.hidden_units}')
        print(f'Learning rate: {self.learning_rate}')
        print(f'Activation: {self.activation}')
        print(f'Optimizer: {self.optimizer}')
        
        # Plot ground truth and prediction
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for idx, point in enumerate(inputs):
            plt.plot(point[0], point[1], 'ro' if labels[idx][0] == 0 else 'bo')
        
        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for idx, point in enumerate(inputs):
            plt.plot(point[0], point[1],
                     'ro' if pred_labels[idx][0] == 0 else 'bo')

        # Plot learning curve
        plt.figure()
        plt.title('Learning curve', fontsize=18)
        plt.plot(self.learning_epoch, self.learning_loss)

        plt.show()

    @staticmethod
    def mse_loss(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        # Mean squared error loss
        return np.mean((prediction - ground_truth) ** 2)

    @staticmethod
    def mse_derivative_loss(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        # Derivative of MSE loss
        return 2 * (prediction - ground_truth) / len(ground_truth)
