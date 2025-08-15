import itertools
from typing import Tuple, Union

import numpy as np
import scipy.signal
from numpy.typing import ArrayLike

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class _LayerInput:
    """
    _LayerInput class represents the input layer of the neural network.
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

        Returns
        -------
        None
        """
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.count_trainable_params = n_neurons * (n_inputs + 1)

        rng = np.random.default_rng()
        self.weights: Float64Array2D = 0.01 * rng.standard_normal((self.n_inputs, self.n_neurons))
        self.biases: Float64Array2D = np.zeros((1, self.n_neurons))
        # L1 strength
        self.weight_regularizer_l1: float = weight_regularizer_l1
        self.bias_regularizer_l1: float = bias_regularizer_l1
        # L2 strength
        self.weight_regularizer_l2: float = weight_regularizer_l2
        self.bias_regularizer_l2: float = bias_regularizer_l2

    def __str__(self):
        return (
            f"Layer_Dense(): n_inputs: {self.n_inputs} "
            f"| n_neurons: {self.n_neurons} "
            f"| L1_w: {self.weight_regularizer_l1} "
            f"| L1_b: {self.bias_regularizer_l1} "
            f"| L2_w: {self.weight_regularizer_l2} "
            f"| L2_b: {self.bias_regularizer_l2}"
            f"| params: {self.count_trainable_params}"
        )

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        """Performs the forward pass of the dense layer.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input data.
        training : bool
            Whether the layer is in training mode (unused in this layer).

        Returns
        -------
        None
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

        Returns
        -------
        None
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

        Parameters
        ----------
        None

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

        Returns
        -------
        None
        """
        self.weights = weights
        self.biases = biases


class LayerDropout:
    """
    Dropout layer for regularization in neural networks.

    This layer randomly sets a fraction of input units to zero during training to help prevent overfitting.
    The fraction of units dropped is determined by the dropout rate provided at initialization.
    """

    def __init__(self, rate: Union[float, int]) -> None:
        """
        Initialize the Dropout layer.

        Parameters
        ----------
        rate : float or int
            The dropout rate, representing the fraction of input units to drop (set to zero) during training.
            Must be between 0 and 1.
        """

        self.rate = 1 - rate  # The probability of keeping a unit active (1 - dropout rate).

    def __str__(self):
        return f"Layer_Dropout(): dropout rate: {1 - self.rate:.3f}"

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        """
        Perform the forward pass for the Dropout layer.

        During training, randomly sets a fraction of input units to zero and scales the remaining units.
        During inference, the input is returned unchanged.

        Parameters
        ----------
        inputs : np.ndarray
            Input data to the layer.
        training : bool
            Whether the layer is in training mode.

        Returns
        -------
        None
        """
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        rng = np.random.default_rng()
        self.binary_mask = (
            rng.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        )  # The mask applied to the input during training to drop units.
        self.output = self.inputs * self.binary_mask

    def backward(self, dvalues: Float64Array2D) -> None:
        """
        Perform the backward pass for the Dropout layer.

        Multiplies the incoming gradient by the same binary mask used during the forward pass.

        Parameters
        ----------
        dvalues : np.ndarray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        None
        """
        self.dinputs = dvalues * self.binary_mask


class LayerConv:
    """
    Implements a 2D convolutional layer for neural networks.

    This layer applies a set of learnable filters to the input data, enabling the extraction of spatial features.
    It supports configurable filter size, stride, padding, and L1/L2 regularization for both weights and biases.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_filters: int,
        filter_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, Tuple[int, int]] = "valid",
        weight_regularizer_l1: float = 0.0,
        weight_regularizer_l2: float = 0.0,
        bias_regularizer_l1: float = 0.0,
        bias_regularizer_l2: float = 0.0,
    ) -> None:
        """
        Initializes a 2D convolutional layer with the specified parameters.

        This constructor sets up the convolutional layer's filter dimensions, stride, padding, weights, and biases.
        It also configures L1 and L2 regularization strengths for both weights and biases.

        Parameters
        ----------
        input_shape : tuple of int
            shape of the input (channels, height, width).
        n_filters : int
            Number of convolutional filters (output channels).
        filter_size : int or tuple of int
            Size of the convolutional filters (height, width).
        stride : int or tuple of int, optional
            Stride of the convolution (default is 1).
        padding : str or tuple of int, optional
            Padding mode ('valid', 'same', or tuple) (default is 'valid').
        weight_regularizer_l1 : float, optional
            L1 regularization strength for weights (default is 0.0).
        weight_regularizer_l2 : float, optional
            L2 regularization strength for weights (default is 0.0).
        bias_regularizer_l1 : float, optional
            L1 regularization strength for biases (default is 0.0).
        bias_regularizer_l2 : float, optional
            L2 regularization strength for biases (default is 0.0).

        Returns
        -------
        None
        """
        self.input_channels, self.input_height, self.input_width = input_shape
        self.n_filters = n_filters

        if isinstance(filter_size, int):
            self.filter_height = self.filter_width = filter_size
        else:
            self.filter_height, self.filter_width = filter_size

        if isinstance(stride, int):
            self.stride_y = self.stride_x = stride
        else:
            self.stride_y, self.stride_x = stride

        if isinstance(padding, str):
            if padding == "same":  # observe - kernel with even dim outputs weird results
                self.pad_y = (self.filter_height - 1) // 2
                self.pad_x = (self.filter_width - 1) // 2
            elif padding == "valid":
                self.pad_x = self.pad_y = 0
            else:
                raise ValueError("padding must be 'valid', 'same', or tuple (int, int)")
        else:
            self.pad_y, self.pad_x = padding

        rng = np.random.default_rng()
        self.weights = 0.01 * rng.standard_normal(
            (self.n_filters, self.input_channels, self.filter_height, self.filter_width),
        )
        self.biases = np.zeros((self.n_filters, 1))

        self.count_trainable_params = self.n_filters * ((self.input_channels * self.filter_height * self.filter_width) + 1)

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(
        self,
        inputs: np.ndarray,
        training: bool,
    ) -> None:
        """
        Performs the forward pass of the 2D convolutional layer.

        Applies the convolution operation to the input data using the layer's filters, stride, and padding.
        The output is computed for each filter and input in the batch.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (batch_size, channels, height, width).
        training : bool
            Whether the layer is in training mode (unused in this layer).

        Returns
        -------
        None
        """
        self.inputs = inputs
        # batch size is number of images in a batch
        batch_size, channels, height, width = inputs.shape

        padded_inputs = np.pad(
            inputs,  # default pad value is zero, only need to pad, 3rd & 4th dim, the images.
            ((0, 0), (0, 0), (self.pad_y, self.pad_y), (self.pad_x, self.pad_x)),
            mode="constant",
        )

        out_height = ((height + 2 * self.pad_y - self.filter_height) // self.stride_y) + 1
        out_width = ((width + 2 * self.pad_x - self.filter_width) // self.stride_x) + 1

        self.output = np.zeros((batch_size, self.n_filters, out_height, out_width))

        ###### utilizing scipy.signal.correlate2d method, optimized for larger inputs and kernels
        for n in range(batch_size):
            for f in range(self.n_filters):
                conv_sum = np.zeros((padded_inputs.shape[2] - self.filter_height + 1, padded_inputs.shape[3] - self.filter_width + 1))
                for c in range(channels):
                    # correlate2d returns the full convolution, so we use 'valid' mode for correct shape
                    conv_sum += scipy.signal.correlate2d(padded_inputs[n, c], self.weights[f, c], mode="valid")
                # Apply stride
                self.output[n, f] = conv_sum[:: self.stride_y, :: self.stride_x][:out_height, :out_width] + self.biases[f]

        ##### Manual Implementation - convolution
        # for i in range(out_height):
        #     for j in range(out_width):
        #         h_start = i * self.stride_y
        #         h_end = h_start + self.filter_height
        #         w_start = j * self.stride_x
        #         w_end = w_start + self.filter_width

        #         region = padded_inputs[:, :, h_start:h_end, w_start:w_end]
        #         for f in range(self.n_filters):
        #             self.output[:, f, i, j] = (
        #                 np.sum(
        #                     region * self.weights[f, :, :, :],
        #                     axis=(1, 2, 3),
        #                 )
        #                 + self.biases[f]
        #             )

    def backward(
        self,
        dvalues: np.ndarray,
    ) -> None:
        """
        Performs the backward pass of the 2D convolutional layer.

        Computes the gradients of the loss with respect to the layer's inputs, weights, and biases.
        This method also applies L1 and L2 regularization to the computed gradients.

        Parameters
        ----------
        dvalues : np.ndarray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        None
        """
        batch_size, channels, height, width = self.inputs.shape
        _, _, out_height, out_width = dvalues.shape

        padded_inputs = np.pad(
            self.inputs,
            ((0, 0), (0, 0), (self.pad_y, self.pad_y), (self.pad_x, self.pad_x)),
            mode="constant",
        )

        dinputs_padded = np.zeros_like(padded_inputs)
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

        ##### utilizing scipy.signal.correlate2d and scipy.signal.convolve2d methods, optimized
        # Gradient w.r.t. biases: sum over batch, height, width
        for f in range(self.n_filters):
            self.dbiases[f] = np.sum(dvalues[:, f])

        # Gradient w.r.t. weights and inputs
        for n in range(batch_size):
            for f, c in itertools.product(range(self.n_filters), range(channels)):
                # dweights: correlate input with dvalues (as in forward, but swap roles)
                self.dweights[f, c] += scipy.signal.correlate2d(
                    padded_inputs[n, c],
                    dvalues[n, f],
                    mode="valid",
                )
                # dinputs: convolve dvalues with weights (full mode for padding)
                dinputs_padded[n, c] += scipy.signal.convolve2d(
                    dvalues[n, f],
                    self.weights[f, c],
                    mode="full",
                )

        # Remove padding from dinputs
        if self.pad_y == 0 and self.pad_x == 0:
            self.dinputs = dinputs_padded
        else:
            self.dinputs = dinputs_padded[:, :, self.pad_y : -self.pad_y or None, self.pad_x : -self.pad_x or None]

        ##### Manual implementation
        # for i in range(out_height):
        #     for j in range(out_width):
        #         h_start = i * self.stride_y
        #         h_end = h_start + self.filter_height
        #         w_start = j * self.stride_x
        #         w_end = w_start + self.filter_width

        #         region = padded_inputs[:, :, h_start:h_end, w_start:w_end]
        #         for f in range(self.n_filters):
        #             self.dweights[f] += np.sum(
        #                 region * dvalues[:, f, i, j][:, None, None, None],
        #                 axis=0,
        #             )
        #             self.biases[f] += np.sum(dvalues[:, f, i, j])

        #         for n in range(batch_size):
        #             for f in range(self.n_filters):
        #                 dinputs_padded[n, :, h_start:h_end, w_start:w_end] += self.weights[f] * dvalues[n, f, i, j]

        #         # Remove padding from inputs
        #         if self.pad_y == 0 and self.pad_x == 0:
        #             self.dinputs = dinputs_padded
        #         else:
        #             self.dinputs = dinputs_padded[:, :, self.pad_y : -self.pad_y or None, self.pad_x : -self.pad_x or None]

        # Regularization
        if self.weight_regularizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dl1
        if self.bias_regularizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dl1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

    def __str__(self) -> str:
        """
        Returns a string representation of the 2D convolutional layer.

        The string includes information about the kernel count, kernel channels, kernel dimensions, stride, and padding.

        Returns
        -------
        str
            A string describing the layer's configuration.
        """
        return (
            f"Layer_Conv2D(): kernel_count:{self.weights.shape[0]} "
            f"| kernel_channels:{self.weights.shape[1]} "
            f"| kernel_dim:({self.weights.shape[2]},{self.weights.shape[3]}) "
            f"| stride:({self.stride_y},{self.stride_x}) "
            f"| padding:({self.pad_y},{self.pad_x})"
            f"| params: {self.count_trainable_params}"
        )

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the parameters (weights and biases) of the 2D convolutional layer.

        This method provides access to the layer's trainable parameters for optimization or inspection.

        Returns
        -------
        tuple
            A tuple containing the weights and biases of the layer.
        """
        return self.weights, self.biases

    def set_parameters(self, weights, biases) -> None:
        """
        Sets the parameters (weights and biases) of the 2D convolutional layer.

        This method updates the layer's weights and biases with the provided values.

        Parameters
        ----------
        weights : np.ndarray
            The new weights to set for the layer.
        biases : np.ndarray
            The new biases to set for the layer.

        Returns
        -------
        None
        """
        self.weights = weights
        self.biases = biases


class LayerMaxPool2D:
    """
    Implements a 2D max pooling layer for neural networks.

    This layer reduces the spatial dimensions of the input by taking the maximum value over a specified window for each channel.
    It is commonly used to downsample feature maps and introduce spatial invariance.
    """

    def __init__(self, pool_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = None) -> None:
        """
        Initializes the max pooling layer with the specified pool size and stride.

        This constructor sets up the pooling window dimensions and stride for the max pooling operation.

        Parameters
        ----------
        pool_size : int or tuple of int
            Size of the pooling window (height, width).
        stride : int or tuple of int, optional
            Stride of the pooling operation. If None, defaults to pool size.

        Returns
        -------
        None
        """
        if isinstance(pool_size, int):
            self.pool_height = self.pool_width = pool_size
        else:
            self.pool_height, self.pool_width = pool_size

        if stride is None:
            self.stride_y = self.pool_height
            self.stride_x = self.pool_width
        elif isinstance(stride, int):
            self.stride_y = self.stride_x = stride
        else:
            self.stride_y, self.stride_x = stride

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Performs the forward pass of the max pooling layer.

        Applies the max pooling operation to the input data, reducing its spatial dimensions by taking the maximum value in each pooling window.
        The indices of the maximum values are stored for use in the backward pass.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (batch_size, channels, height, width).
        training : bool
            Whether the layer is in training mode (unused in this layer).

        Returns
        -------
        None
        """
        self.inputs = inputs
        batch_size, channels, height, width = inputs.shape

        out_height = (height - self.pool_height) // self.stride_y + 1
        out_width = (width - self.pool_width) // self.stride_x + 1

        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros(self.output.shape + (2,), dtype=np.int32)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride_y
                h_end = h_start + self.pool_height
                w_start = j * self.stride_x
                w_end = w_start + self.pool_width

                region = inputs[:, :, h_start:h_end, w_start:w_end]
                # region shape: (batch_size, channels, pool_height, pool_width)
                region_reshaped = region.reshape(batch_size, channels, -1)
                max_idx = np.argmax(region_reshaped, axis=2)
                max_val = np.max(region_reshaped, axis=2)
                self.output[:, :, i, j] = max_val

                # Store the indices for backward pass
                max_row = max_idx // self.pool_width
                max_col = max_idx % self.pool_width
                self.max_indices[:, :, i, j, 0] = max_row
                self.max_indices[:, :, i, j, 1] = max_col

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Performs the backward pass of the max pooling layer.

        Computes the gradient of the loss with respect to the input of the max pooling layer.
        The gradient is propagated only to the positions of the maximum values selected during the forward pass.

        Parameters
        ----------
        dvalues : np.ndarray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        None
        """
        batch_size, channels, height, width = self.inputs.shape
        _, _, out_height, out_width = dvalues.shape

        self.dinputs = np.zeros_like(self.inputs)

        for i, j in itertools.product(range(out_height), range(out_width)):
            h_start = i * self.stride_y
            w_start = j * self.stride_x

            max_row = self.max_indices[:, :, i, j, 0]
            max_col = self.max_indices[:, :, i, j, 1]

            for n, c in itertools.product(range(batch_size), range(channels)):
                self.dinputs[n, c, h_start + max_row[n, c], w_start + max_col[n, c]] += dvalues[n, c, i, j]

    def __str__(self):
        """Returns a string representation of the max pooling layer.

        The string includes information about the pooling window size and stride.

        Returns
        -------
        str
            A string describing the layer's configuration.
        """
        return f"Layer_MaxPool2D(): pool_size: ({self.pool_height},{self.pool_width}) | stride: ({self.stride_y},{self.stride_x})"


class LayerFlatten:
    """
    Implements a flattening layer for CNN architectures.

    This layer reshapes multi-dimensional input (e.g., from convolutional or pooling layers)
    into a 2D array suitable for dense layers: (batch_size, -1).
    """

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Flattens the input to shape (batch_size, -1).

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (batch_size, channels, height, width).
        training : bool
            Whether the layer is in training mode (unused).

        Returns
        -------
        None
        """
        self.inputs_shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Reshapes the gradient to the original input shape.

        Parameters
        ----------
        dvalues : np.ndarray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        None
        """
        self.dinputs = dvalues.reshape(self.inputs_shape)

    def __str__(self):
        return "Layer_Flatten()"


LayerTypes = Union[LayerDense, LayerDropout]
TrainableLayerTypes = LayerDense
