from typing import Tuple

import numpy as np

from netweaver.activation_layers import ActivationSoftmax
from netweaver.lossfunctions import LossCategoricalCrossentropy

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class ActivationSoftmaxLossCategoricalCrossentropy:
    """
    #### what
        - Softmax classifier is a combination of softmax activation and categorical cross-entropy loss.
        - peforms backward pass in a single step faster than traditional (softmax and categorical cross-entropy loss) backward methods.
        - gradients of loss functions with respect to the penultimate layer's outputs reduced to a single step.
    #### Improve
    #### Flow
        - [init -> forward -> backward]
        - formula = predicted values - true values
    """

    def __init__(self) -> None:
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs: Float64Array2D, y_true) -> np.float64:
        """
        #### Note
            - performs forward method of softmax and loss classes.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues: Float64Array2D, y_true) -> None:
        """
        #### Note
            - expects y_true to be sparse labels
            - copy the dvalues to self.dinputs
            - compute gradient and normalize them
        """
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
