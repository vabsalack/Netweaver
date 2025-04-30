from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class LayerInput:
    """
    LayerInput class represents the input layer of the neural network.
    """

    def forward(self, inputs: ArrayLike, training: bool) -> None:
        """
        Performs a forward pass of the input layer.

        #### Parameters:
        inputs (NDArray): Input data.
        """
        self.output = np.array(inputs, dtype=np.float64)


class LayerDense:
    """
    A dense layer implementation.

    This layer performs a linear transformation of the input data followed by a bias addition.
    It supports L1 and L2 regularization for both weights and biases.
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        weight_regularizer_l1: float = 0.0,
        weight_regularizer_l2: float = 0.0,
        bias_regularizer_l1: float = 0.0,
        bias_regularizer_l2: float = 0.0,
    ) -> None:
        """
        Initializes the dense layer with random weights and zero biases.

        Parameters
        ----------
        n_inputs : int
            Number of input features
        n_neurons : int
            Number of neurons in the layer
        weight_regularizer_l1 : float, default=0.0
            L1 regularization strength for weights
        weight_regularizer_l2 : float, default=0.0
            L2 regularization strength for weights
        bias_regularizer_l1 : float, default=0.0
            L1 regularization strength for biases
        bias_regularizer_l2 : float, default=0.0
            L2 regularization strength for biases
        """
        rng = np.random.default_rng()
        self.weights: Float64Array2D = 0.01 * rng.standard_normal((n_inputs, n_neurons))
        self.biases: Float64Array2D = np.zeros((1, n_neurons))
        # L1 strength
        self.weight_regularizer_l1: float = weight_regularizer_l1
        self.bias_regularizer_l1: float = bias_regularizer_l1
        # L2 strength
        self.weight_regularizer_l2: float = weight_regularizer_l2
        self.bias_regularizer_l2: float = bias_regularizer_l2

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        """Performs the forward pass of the dense layer.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input data.
        training : bool
            Whether the layer is in training mode (unused in this layer).
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: Float64Array2D) -> None:
        """Performs the backward pass of the dense layer and computes the dweights, dbiases and dinputs.
        This method also incoporates L1 and L2 regularization to the computed gradients.
        Here the derivate of Absolute function is consider **1 for 0** and positive values, and -1 for negative values.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Gradients of the loss with respect to the layer's output.
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        # apply L1
        if self.weight_regularizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dl1
        if self.bias_regularizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dl1
        # apply L2
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

    def get_parameters(self) -> Tuple[Float64Array2D, Float64Array2D]:
        """Returns the layer's parameters (weights and biases).

        Returns
        -------
        tuple
            A tuple containing the weights and biases.
        """
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        """Sets the layer's parameters (weights and biases).

        Parameters
        ----------
        weights : numpy.ndarray
            Weights to set.
        biases : numpy.ndarray
            Biases to set.
        """
        self.weights = weights
        self.biases = biases


class LayerDropout:
    """
    #### what
        - Dropout is a regularization technique where randomly selected neurons are ignored during training.
        - args: rate (percentage of neurons to be deactivated)
    #### Improve
    #### Flow
        - [init -> (forward -> backward)]
        - create binary mask from sample
    """

    def __init__(self, rate: Union[float, int]) -> None:
        """
        - rate = 1-rate # np.random.binomial expects probability of success (1) not failure (0)
        """
        self.rate = 1 - rate

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        """
        - divide the mask by the rate to scale the values.
        """
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        rng = np.random.default_rng()
        self.binary_mask = rng.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        self.output = self.inputs * self.binary_mask

    def backward(self, dvalues: Float64Array2D) -> None:
        self.dinputs = dvalues * self.binary_mask


LayerTypes = Union[LayerDense, LayerDropout]
TrainableLayerTypes = LayerDense
