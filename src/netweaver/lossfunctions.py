from typing import List, Tuple, Union

import numpy as np

from netweaver.layers import TrainableLayerTypes

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class Loss:
    """
    #### what
        - Base class for loss functions.
    #### Improve
        - Implement additional loss functions as needed.
    ### Flow
        - [remember_trainable_layers -> regularization_loss -> calculate]
        - remember_trainable_layers method sets self.trainable_layers property.
        - regularization_loss method calculates the regularization loss using layers in self.trainable_layers.
        - calculate method calculates the data loss using child class's forward method and regularization loss from regularization_loss method.
    """

    def remember_trainable_layers(
        self, trainable_layers: List[TrainableLayerTypes]
    ) -> None:
        self.trainable_layers = trainable_layers

    def regularization_loss(self) -> float:
        regularization_loss = 0.0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_L1 > 0:  # L1
                regularization_loss += layer.weight_regularizer_L1 * np.sum(
                    np.abs(layer.weights)
                )
            if layer.bias_regularizer_L1 > 0:
                regularization_loss += layer.bias_regularizer_L1 * np.sum(
                    np.abs(layer.biases)
                )
            if layer.weight_regularizer_L2 > 0:  # L2
                regularization_loss += layer.weight_regularizer_L2 * np.sum(
                    layer.weights**2
                )
            if layer.bias_regularizer_L2 > 0:
                regularization_loss += layer.bias_regularizer_L2 * np.sum(
                    layer.biases**2
                )
        return regularization_loss

    def calculate(
        self,
        output: Float64Array2D,
        y,  # y type can be one-hot encoded or sparse lables
        *,
        include_regularization: bool = False,
    ) -> Union[np.float64, Tuple[np.float64, np.float64]]:
        """
        #### Note
            - data loss is the mean of sample losses
            - regularization loss is the sum of L1 and L2 penalties
            - returns only data loss by default
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def calculate_accumulated(
        self, *, include_regularization: bool = False
    ) -> Union[float, Tuple[np.float64, np.float64]]:
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def new_pass(self) -> None:
        self.accumulated_sum = 0
        self.accumulated_count = 0


class LossCategoricalCrossentropy(Loss):
    """Implements the categorical cross-entropy loss function, commonly used for multi-class classification tasks. 
    This loss function measures the dissimilarity between the predicted probability distribution and the true 
    distribution by calculating the negative log-likelihood of the true class probabilities given the predicted 
    probabilities. It is minimized when the predicted probabilities align closely with the true labels, effectively 
    finding the Maximum Likelihood Estimation (MLE) of the model. This class provides methods for both forward 
    computation of the loss and backward computation of the gradient for optimization purposes.
    """

    def forward(
        self, y_pred: Float64Array2D, y_true: np.ndarray
    ) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        # sourcery skip: inline-immediately-returned-variable
        """Calculates the categorical cross-entropy loss.
        Clips the predicted values to prevent division by zero.
        Clips both sides to not drag mean towards any value.

        Parameters
        ----------
        y_pred : numpy.ndarray
            Predicted probabilities.
        y_true : numpy.ndarray
            True labels, either one-hot encoded or as a vector of integers.

        Returns
        -------
        numpy.ndarray
            The negative log-likelihoods for each sample.
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true, axis=1, keepdims=False
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues: Float64Array2D, y_true: np.ndarray) -> None:
        """Calculates the gradient of the categorical cross-entropy loss and normalizes it by the number of samples.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Predicted probabilities.
        y_true : numpy.ndarray
            True labels, either one-hot encoded or as a vector of integers.
        """
        samples = len(dvalues)
        if len(y_true.shape) == 1:
            labels = len(dvalues[0])
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class LossBinaryCrossentropy(Loss):
    """
    #### what
        - Binary cross-entropy loss function is used in binary regression models.
        - used primarily in the multi-label classification problem.
    #### Improve
    #### Flow
        - [forward -> backward]
        - formula: avg(-sum(yi_true * log(yi_pred) + (1 - yi_true) * log(1 - yi_pred)))
    """

    def forward(
        self,
        y_pred: Float64Array2D,
        y_true: np.ndarray[Tuple[int, int], np.dtype[np.int64]],
    ) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        """
        #### Note
            - Unlike categorical cross-entropy, it measures negative-log-likelihood of each output neurons seperately and **average** them.
            - Each sample loss is a vector of output neuron's losses and it's average used as the final loss for that sample.
        """
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        sample_losses = np.mean(sample_losses, axis=1)
        return sample_losses

    def backward(
        self,
        dvalues: Float64Array2D,
        y_true: np.ndarray[Tuple[int, int], np.dtype[np.int64]],
    ) -> None:
        """
        #### Note
            - ∂L/∂y^ =  -((y_true / y_pred)-(1 - y_true)/(1 - y_pred))/outputs.
            - Normalize the gradient by the number of samples in the batch.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = (
            -((y_true / clipped_dvalues) - ((1 - y_true) / (1 - clipped_dvalues)))
            / outputs
        )
        self.dinputs /= samples


class LossMeanSquaredError(Loss):
    """
    #### What
        - Mean squared error(MSE) loss function is used in single or multiple output regression tasks.
        - MSE(L2) most commonly used loss function for regression tasks over MAE(L1).
        - Assumption is that the error residuals are normally distributed.
    #### Improve
    #### Flow
        - [forward -> backward]
        - formula: mean((y_true - y_pred)^2)
    """

    def forward(
        self, y_pred: Float64Array2D, y_true: Float64Array2D
    ) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        # sourcery skip: inline-immediately-returned-variable
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues: Float64Array2D, y_true: Float64Array2D) -> None:
        """
        #### Note
            - ∂L/∂y^ = -2 * (y_true - y_pred) / outputs =  for one of predicted outputs in a sample.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = (
            -2 * (y_true - dvalues) / outputs
        )  # wonder log and negative missing. it's cancelled out. look into MLE.
        self.dinputs = self.dinputs / samples


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

    def forward(
        self, y_pred: Float64Array2D, y_true: Float64Array2D
    ) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        # sourcery skip: inline-immediately-returned-variable
        """
        #### Note
            - loss formula: mean(abs(y_true - y_pred))
        """
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues: Float64Array2D, y_true: Float64Array2D) -> None:
        """
        #### Note
            - ∂L/∂y^ = -sign(y_true - y_pred) / outputs = for one of predicted outputs in a sample.
            - It's negative of sign(y_true - y_pred), graph of y_pred > y_true is +1 and y_pred < y_true is -1
            - normalize the gradient by the number of samples in the batch.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = (
            -np.sign(y_true - dvalues) / outputs
        )  # np.sign returns -1 (<0) 0 (=0) 1 (>0)
        self.dinputs = self.dinputs / samples


LossTypes = Union[
    LossBinaryCrossentropy,
    LossCategoricalCrossentropy,
    LossMeanSquaredError,
    LossMeanAbsoluteError,
]
