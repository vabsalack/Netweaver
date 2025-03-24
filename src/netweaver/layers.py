from netweaver.utils import ArrayLike, Float64Array2D, Union, np


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
    #### what
        - create Dense Layer
        - args: nintputs, nneurons, wL1, wL2, bL1, bL2
    #### Improve
        - experiment with other initialization method such as he, xavier, etc.
    #### Flow
        - [init -> (forward -> backward)]
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        weight_regularizer_L1: Union[float, int] = 0,
        weight_regularizer_L2: Union[float, int] = 0,
        bias_regularizer_L1: Union[float, int] = 0,
        bias_regularizer_L2: Union[float, int] = 0,
    ) -> None:
        """
        #### Note
            - weights initialization is one of crucial part in model convergence.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # L1 strength
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.bias_regularizer_L1 = bias_regularizer_L1
        # L2 strength
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L2 = bias_regularizer_L2

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: Float64Array2D) -> None:
        """
        #### Note
        - compute gradients.
        - apply regularization to computed gradients.
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        # apply L1
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1
        # apply L2
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
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
        self.binary_mask = (
            np.random.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        )
        self.output = self.inputs * self.binary_mask

    def backward(self, dvalues: Float64Array2D) -> None:
        self.dinputs = dvalues * self.binary_mask
