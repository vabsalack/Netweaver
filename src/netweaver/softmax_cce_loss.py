from typing import Tuple

import numpy as np

from .activation_layers import ActivationSoftmax
from .lossfunctions import LossCategoricalCrossentropy

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class ActivationSoftmaxLossCategoricalCrossentropy:
    """Combines softmax activation and categorical cross-entropy loss.

    Provides an optimized implementation for calculating the loss graidents w.r.t penultimate layer outputs in a single step faster than traditional\
          softmax and categorical loss backward methods.
    Formula = predicted values - true values
    """

    def __init__(self) -> None:
        """
        Initializes the combined softmax activation and categorical cross-entropy loss object.

        This constructor sets up the internal softmax activation and loss function for efficient forward and backward passes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs: Float64Array2D, y_true) -> np.float64:
        """Performs the forward pass, calculating the combined loss.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input data.
        y_true : numpy.ndarray
            True labels.

        Returns
        -------
        numpy.float64
            The calculated loss.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues: Float64Array2D, y_true) -> None:
        """Calculates the gradient of the combined loss w.r.t penultimate layer output.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Input data.
        y_true : numpy.ndarray
            True labels.

        Returns
        -------
        None
        """
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
