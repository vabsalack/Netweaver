from typing import Tuple, Union

import numpy as np

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class ActivationReLU:
    """
    #### what
        - introduce non-linearity in the network.
    #### Improve
        - leaky ReLu, Parametric ReLu, Exponential ReLu, etc.
    #### Flow
        - [(forward -> backward), predictions]
        - ReLu(x) = max(0, x)
    """

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)

    def backward(self, dvalues: Float64Array2D) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs: Float64Array2D) -> Float64Array2D:
        return outputs


class ActivationSoftmax:
    """
    #### what
        - convert raw scores into probabilities.
        - use in the output layer of a neural network for **multi-class** classification.
    #### Improve
    ### Flow
        - [(forward -> backward), predictions]
        - softmax(x) = exp(x) / sum(exp(x))
        - pair it with compatible loss function such as categorical cross-entropy loss.
    """

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(self.inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues: Float64Array2D) -> None:
        """
        #### How
            - softmax derivative is calculated using jacobian matrix (square matrix)
        """
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)  # here the jacobian is square and symmentrix matrix
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs: Float64Array2D) -> np.ndarray[Tuple[int], np.dtype[np.int64]]:
        return np.argmax(outputs, axis=1)


class ActivationSigmoid:
    """
    #### What
        - use primarily in the multi-label classification problem. Each output neurons represents seperate class on its own.
        - use in the output layer of a neural network for binary logistic regression models
    #### Improve
        - Accuracy
            - Use subset accuracy when exact matches are critical.
            - For tasks where partial matches are acceptable(default here), consider using Hamming Loss, F1 Score, or Jaccard Similarity.
    ### Flow
        - [(forward -> backward), predictions]
        - sigmoid(x) = 1 / (1 + exp(-x))
        - pair it with compatible loss function such as binary cross-entropy loss.
    """

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: Float64Array2D) -> None:
        """
        - derivate: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        """
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs: Float64Array2D) -> np.ndarray[Tuple[int, int], np.dtype[np.int64]]:
        """
        - product of arr: NDArray[bool_] and x: int is y: NDArray[int64]
        - array([False, True]) * -2 = array([0, -2])
        """
        return (outputs > 0.5) * 1


class ActivationLinear:
    """
    #### what
        - Linear activation function is used to pass the input directly to the output without any modification.
        - use in the output layer for regression tasks where we need to predict continuous values.
    #### Improve
    ### Flow
        - [(forward -> backward), predictions]
        - f(x) = x
        - pair it with compatible loss function such as mean squared/absolute error loss.
    """

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues: Float64Array2D) -> None:
        self.dinputs = dvalues.copy()

    def predictions(self, outputs: Float64Array2D) -> Float64Array2D:
        return outputs


ActivationTypes = Union[ActivationSoftmax, ActivationLinear, ActivationReLU, ActivationSigmoid]
