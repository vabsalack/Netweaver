from typing import Tuple, Union

import numpy as np

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class ActivationReLU:
    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        """
        Performs the forward pass of the ReLU activation function.

        Parameters
        ----------
        inputs : Float64Array2D
            The input data for the activation function.
        training : bool
            Indicates whether the layer is in training mode.

        Returns
        -------
        None
        """
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)

    def backward(self, dvalues: Float64Array2D) -> None:
        """
        Performs the backward pass of the ReLU activation function.

        This method computes the gradient of the loss with respect to the inputs for the ReLU activation.

        Parameters
        ----------
        dvalues : Float64Array2D
            The gradient of the loss with respect to the output of this layer.

        Returns
        -------
        None
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs: Float64Array2D) -> Float64Array2D:
        """
        Returns the predictions for the ReLU activation layer.

        This method simply returns the outputs as-is, since ReLU does not modify predictions during inference.

        Parameters
        ----------
        outputs : Float64Array2D
            The output data from the activation function.

        Returns
        -------
        Float64Array2D
            The predictions, which are the same as the input outputs.
        """
        return outputs

    def __str__(self):
        return "Activation_ReLu()"


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
        """
        Performs the forward pass of the Softmax activation function.

        This method converts raw scores into probabilities for multi-class classification tasks.

        Parameters
        ----------
        inputs : Float64Array2D
            The input data for the activation function.
        training : bool
            Indicates whether the layer is in training mode.

        Returns
        -------
        None
        """
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(self.inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues: Float64Array2D) -> None:
        """
        Performs the backward pass of the Softmax activation function.

        This method computes the gradient of the loss with respect to the inputs using the Jacobian matrix for each sample.

        Parameters
        ----------
        dvalues : Float64Array2D
            The gradient of the loss with respect to the output of this layer.

        Returns
        -------
        None
        """
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)  # here the jacobian is square and symmentrix matrix
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs: Float64Array2D) -> np.ndarray[Tuple[int], np.dtype[np.int64]]:
        """
        Returns the predicted class indices for the Softmax activation layer.

        This method computes the index of the maximum probability for each sample, representing the predicted class.

        Parameters
        ----------
        outputs : Float64Array2D
            The output probabilities from the Softmax activation function.

        Returns
        -------
        np.ndarray[Tuple[int], np.dtype[np.int64]]
            The predicted class indices for each sample.
        """
        return np.argmax(outputs, axis=1)

    def __str__(self):
        return "Activation_Softmax()"


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
        """
        Performs the forward pass of the Sigmoid activation function.

        This method applies the sigmoid function to the input data, producing outputs between 0 and 1.

        Parameters
        ----------
        inputs : Float64Array2D
            The input data for the activation function.
        training : bool
            Indicates whether the layer is in training mode.

        Returns
        -------
        None
            This method does not return a value.
        """
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: Float64Array2D) -> None:
        """
        Performs the backward pass of the Sigmoid activation function.

        This method computes the gradient of the loss with respect to the inputs for the Sigmoid activation.

        Parameters
        ----------
        dvalues : Float64Array2D
            The gradient of the loss with respect to the output of this layer.

        Returns
        -------
        None
        """
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs: Float64Array2D) -> np.ndarray[Tuple[int, int], np.dtype[np.int64]]:
        """
        Returns the predicted class labels for the Sigmoid activation layer.

        This method applies a threshold of 0.5 to the outputs, returning 1 for values above 0.5 and 0 otherwise.

        Parameters
        ----------
        outputs : Float64Array2D
            The output data from the Sigmoid activation function.

        Returns
        -------
        np.ndarray[Tuple[int, int], np.dtype[np.int64]]
            The predicted class labels for each sample.
        """
        return (outputs > 0.5) * 1

    def __str__(self):
        return "Activation_Sigmoid()"


class ActivationLinear:
    # """
    # #### what
    #     - Linear activation function is used to pass the input directly to the output without any modification.
    #     - use in the output layer for regression tasks where we need to predict continuous values.
    # #### Improve
    # ### Flow
    #     - [(forward -> backward), predictions]
    #     - f(x) = x
    #     - pair it with compatible loss function such as mean squared/absolute error loss.
    # """

    def forward(self, inputs: Float64Array2D, training: bool) -> None:
        """
        Performs the forward pass of the Linear activation function.

        This method passes the input directly to the output without any modification.

        Parameters
        ----------
        inputs : Float64Array2D
            The input data for the activation function.
        training : bool
            Indicates whether the layer is in training mode.

        Returns
        -------
        None
        """
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues: Float64Array2D) -> None:
        """
        Performs the backward pass of the Linear activation function.

        This method copies the gradient of the loss with respect to the output directly to the input.

        Parameters
        ----------
        dvalues : Float64Array2D
            The gradient of the loss with respect to the output of this layer.

        Returns
        -------
        None
        """
        self.dinputs = dvalues.copy()

    def predictions(self, outputs: Float64Array2D) -> Float64Array2D:
        """
        Returns the predictions for the Linear activation layer.

        This method simply returns the outputs as-is, since the linear activation does not modify predictions.

        Parameters
        ----------
        outputs : Float64Array2D
            The output data from the activation function.

        Returns
        -------
        Float64Array2D
            The predictions, which are the same as the input outputs.
        """
        return outputs

    def __str__(self):
        """
        Returns a string representation of the ActivationLinear object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return "Activation_Linear()"


ActivationTypes = Union[ActivationSoftmax, ActivationLinear, ActivationReLU, ActivationSigmoid]
