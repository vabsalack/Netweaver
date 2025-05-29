from typing import List, Tuple, Union

import numpy as np

from .layers import TrainableLayerTypes

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class Loss:
    # """
    # #### what
    #     - Base class for loss functions.
    # #### Improve
    #     - Implement additional loss functions as needed.
    # ### Flow
    #     - [remember_trainable_layers -> regularization_loss -> calculate]
    #     - remember_trainable_layers method sets self.trainable_layers property.
    #     - regularization_loss method calculates the regularization loss using layers in self.trainable_layers.
    #     - calculate method calculates the data loss using child class's forward method and regularization loss from regularization_loss method.
    # """

    def remember_trainable_layers(self, trainable_layers: List[TrainableLayerTypes]) -> None:
        """
        Stores the list of trainable layers for use in regularization loss calculation.

        Parameters
        ----------
        trainable_layers : List[TrainableLayerTypes]
            The list of trainable layers in the model.

        Returns
        -------
        None
        """
        self.trainable_layers = trainable_layers

    def regularization_loss(self) -> float:
        """
        Calculates the total regularization loss from all trainable layers.

        This method sums the L1 and L2 regularization losses for weights and biases across all trainable layers.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The total regularization loss.
        """
        regularization_loss = 0.0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:  # L1
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.weight_regularizer_l2 > 0:  # L2
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)
        return regularization_loss

    def calculate(
        self,
        output: Float64Array2D,
        y,  # y type can be one-hot encoded or sparse lables
        *,
        include_regularization: bool = False,
    ) -> Union[np.float64, Tuple[np.float64, np.float64]]:
        """
        Calculates the data loss and optionally the regularization loss.

        This method computes the mean data loss using the child class's forward method and, if requested, adds the regularization loss.

        Parameters
        ----------
        output : Float64Array2D
            The predicted outputs from the model.
        y : numpy.ndarray
            The true labels, which can be one-hot encoded or sparse labels.
        include_regularization : bool, optional
            Whether to include the regularization loss in the output. Defaults to False.

        Returns
        -------
        np.float64 or Tuple[np.float64, np.float64]
            The data loss, or a tuple of (data loss, regularization loss) if include_regularization is True.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization: bool = False) -> Union[float, Tuple[np.float64, np.float64]]:
        """
        Calculates the accumulated data loss and optionally the regularization loss.

        This method computes the mean of all accumulated sample losses and, if requested, adds the regularization loss.

        Parameters
        ----------
        include_regularization : bool, optional
            Whether to include the regularization loss in the output. Defaults to False.

        Returns
        -------
        float or Tuple[np.float64, np.float64]
            The accumulated data loss, or a tuple of (data loss, regularization loss) if include_regularization is True.
        """
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def new_pass(self) -> None:
        """
        Resets the accumulated sum and count for a new pass.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0


class LossCategoricalCrossentropy(Loss):
    """Implements the categorical cross-entropy loss function, commonly used for multi-class classification tasks.
    This loss function measures the dissimilarity between the predicted probability distribution and the true
    distribution by calculating the negative log-likelihood of the true class probabilities given the predicted
    probabilities. It is minimized when the predicted probabilities align closely with the true labels, effectively
    finding the Maximum Likelihood Estimation (MLE) of the model.
    """

    def forward(self, y_pred: Float64Array2D, y_true: np.ndarray) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        """
        Calculates the categorical cross-entropy loss.

        This method computes the negative log-likelihoods for each sample, measuring the dissimilarity between the predicted and true probability \
        distributions.

        Parameters
        ----------
        y_pred : Float64Array2D
            Predicted probabilities.
        y_true : np.ndarray
            True labels, either one-hot encoded or as a vector of integers.

        Returns
        -------
        np.ndarray[Tuple[int], np.dtype[np.float64]]
            The negative log-likelihoods for each sample.
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1, keepdims=False)
        return -np.log(correct_confidences)  # negative_log_likelihoods

    def backward(self, dvalues: Float64Array2D, y_true: np.ndarray) -> None:
        """
        Calculates the gradient of the categorical cross-entropy loss and normalizes it by the number of samples.

        This method computes the gradient of the loss with respect to the predicted probabilities.

        Parameters
        ----------
        dvalues : Float64Array2D
            Predicted probabilities.
        y_true : np.ndarray
            True labels, either one-hot encoded or as a vector of integers.

        Returns
        -------
        None
        """
        samples = len(dvalues)
        if len(y_true.shape) == 1:
            labels = len(dvalues[0])
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

    def __str__(self):
        """
        Returns a string representation of the LossCategoricalCrossentropy object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return "Loss_CategoricalCrossEntropy()"


class LossBinaryCrossentropy(Loss):
    """
    Implements the binary cross-entropy loss function, which is widely used for binary and multi-label classification tasks.

    This loss function quantifies the difference between the predicted probabilities and the true binary labels for each output neuron.
    It is particularly suitable for problems where each output can be independently classified as 0 or 1, such as multi-label classification.

    The binary cross-entropy loss is computed as the average negative log-likelihood across all output neurons and samples:
        loss = mean(-[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)])

    The loss is minimized when the predicted probabilities closely match the true binary labels, making it effective for training models
    that output probabilities for each class independently.
    """

    def forward(
        self,
        y_pred: Float64Array2D,
        y_true: np.ndarray[Tuple[int, int], np.dtype[np.int64]],
    ) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        """
        Calculates the binary cross-entropy loss for each sample.

        This method computes the negative log-likelihood for each output neuron separately and averages them for each sample.

        Parameters
        ----------
        y_pred : Float64Array2D
            Predicted probabilities for each output neuron.
        y_true : np.ndarray[Tuple[int, int], np.dtype[np.int64]]
            True binary labels for each output neuron.

        Returns
        -------
        np.ndarray[Tuple[int], np.dtype[np.float64]]
            The average negative log-likelihood loss for each sample.
        """
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses, axis=1)  # sample_losses

    def backward(
        self,
        dvalues: Float64Array2D,
        y_true: np.ndarray[Tuple[int, int], np.dtype[np.int64]],
    ) -> None:
        """
        Computes the gradient of the binary cross-entropy loss with respect to the predicted values.

        This method normalizes the gradient by the number of samples and outputs, preparing it for backpropagation.
        - ∂L/∂y^ =  -((y_true / y_pred)-(1 - y_true)/(1 - y_pred))/outputs.

        Parameters
        ----------
        dvalues : Float64Array2D
            Predicted probabilities for each output neuron.
        y_true : np.ndarray[Tuple[int, int], np.dtype[np.int64]]
            True binary labels for each output neuron.

        Returns
        -------
        None
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -((y_true / clipped_dvalues) - ((1 - y_true) / (1 - clipped_dvalues))) / outputs
        self.dinputs /= samples

    def __str__(self):
        """
        Returns a string representation of the LossBinaryCrossentropy object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return "Loss_BinaryCrossEntropy()"


class LossMeanSquaredError(Loss):
    """
    Implements the Mean Squared Error (MSE) loss function, widely used for regression tasks with single or multiple outputs.

    - MSE (L2 loss) measures the average squared difference between predicted and true values, penalizing larger errors more heavily than smaller ones
    - It is the most common loss function for regression, preferred over Mean Absolute Error (MAE) when outliers are not a major concern.
    - The MSE loss assumes that the residuals (errors) are normally distributed, making it suitable for many real-world regression problems.

    Usage Flow:
        - [forward -> backward]
        - The forward method computes the mean squared error for each sample: mean((y_true - y_pred)^2).
        - The backward method calculates the gradient of the loss with respect to the predictions, normalized by the number of outputs and samples.

    Improvements:
        - Consider robust alternatives for datasets with significant outliers.
        - Can be extended to support weighted errors or other regression variants.
    """

    def forward(self, y_pred: Float64Array2D, y_true: Float64Array2D) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        """
        Calculates the mean squared error (MSE) loss for each sample.

        This loss function measures the average squared difference between the predicted and true values for each sample.

        Parameters
        ----------
        y_pred : Float64Array2D
            Predicted values for each output.
        y_true : Float64Array2D
            True values for each output.

        Returns
        -------
        np.ndarray[Tuple[int], np.dtype[np.float64]]
            The mean squared error loss for each sample.
        """
        return np.mean((y_true - y_pred) ** 2, axis=-1)  # sample_losses

    def backward(self, dvalues: Float64Array2D, y_true: Float64Array2D) -> None:
        """
        Computes the gradient of the mean squared error (MSE) loss with respect to the predicted values.

        This method normalizes the gradient by the number of samples and outputs, preparing it for backpropagation.

        Parameters
        ----------
        dvalues : Float64Array2D
            Predicted values for each output.
        y_true : Float64Array2D
            True values for each output.

        Returns
        -------
        None
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs  # wonder log and negative missing. it's cancelled out. look into MLE.
        self.dinputs = self.dinputs / samples

    def __str__(self):
        """
        Returns a string representation of the LossMeanSquaredError object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return "Loss_MeanSquaredError()"


class LossMeanAbsoluteError(Loss):
    """
    #### What
        - Mean absolute error(MAE) loss function is used in single or multiple output regression tasks.
        - MAE(L1) is less sensitive to outliers than MSE(L2).
    #### Improve
    #### Flow
        - [forward -> backward]
        - Assumption is that the error residuals are Laplace distributed.
    """

    def forward(self, y_pred: Float64Array2D, y_true: Float64Array2D) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        """
        Calculates the mean absolute error (MAE) loss for each sample.

        This loss function measures the average absolute difference between the predicted and true values for each sample.

        Parameters
        ----------
        y_pred : Float64Array2D
            Predicted values for each output.
        y_true : Float64Array2D
            True values for each output.

        Returns
        -------
        np.ndarray[Tuple[int], np.dtype[np.float64]]
            The mean absolute error loss for each sample.
        """
        return np.mean(np.abs(y_true - y_pred), axis=-1)  # sample_losses

    def backward(self, dvalues: Float64Array2D, y_true: Float64Array2D) -> None:
        """
        Computes the gradient of the mean absolute error (MAE) loss with respect to the predicted values.

        This method normalizes the gradient by the number of samples and outputs, preparing it for backpropagation.
        - ∂L/∂y^ = -sign(y_true - y_pred) / outputs = for one of predicted outputs in a sample.
        - It's negative of sign(y_true - y_pred), graph of y_pred > y_true is +1 and y_pred < y_true is -1
        Parameters
        ----------
        dvalues : Float64Array2D
            Predicted values for each output.
        y_true : Float64Array2D
            True values for each output.

        Returns
        -------
        None
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -np.sign(y_true - dvalues) / outputs  # np.sign returns -1 (<0) 0 (=0) 1 (>0)
        self.dinputs = self.dinputs / samples

    def __str__(self):
        """
        Returns a string representation of the LossMeanAbsoluteError object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return "Loss_MeanAbsoluteError()"


LossTypes = Union[
    LossBinaryCrossentropy,
    LossCategoricalCrossentropy,
    LossMeanSquaredError,
    LossMeanAbsoluteError,
]
