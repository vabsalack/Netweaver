from typing import Union

import numpy as np


class Accuracy:
    def calculate(self, predictions, y):
        """Calculates the accuracy given predictions and true labels.

        Parameters
        ----------
        predictions : numpy.ndarray
            The predicted labels.
        y : numpy.ndarray
            The true labels.

        Returns
        -------
        float
            The calculated accuracy.
        """
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)  # taken boolean False as 0
        self.accumulated_sum += np.sum(comparisons)  # taken boolean sum as 1, sums only true
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        """Calculates the accumulated accuracy.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The accumulated accuracy.
        """
        return self.accumulated_sum / self.accumulated_count

    def new_pass(self):
        """Resets the accumulated sum and count.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0


class AccuracyCategorical(Accuracy):
    """Calculates the accuracy for categorical data.

    Used for classification tasks where predictions are categorical labels.  Compares predicted labels to true labels to determine accuracy.
    """

    def init(self, y):
        """Dummy method for compatibility with the AccuracyRegression class in model's workflow.

        Parameters
        ----------
        y : numpy.ndarray
            The true labels.

        Returns
        -------
        None
        """
        pass

    def compare(self, predictions, y):
        """Compares predictions to true labels.

        Parameters
        ----------
        predictions : numpy.ndarray
            The predicted labels in one-hot encoded format.
        y : numpy.ndarray
            The true labels.

        Returns
        -------
        numpy.ndarray
            A boolean array indicating whether each prediction is correct.
        """
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

    def __str__(self):
        """
        Returns a string representation of the AccuracyCategorical object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The string representation of the object.
        """
        return "Accuracy_Categorical()"


class AccuracyRegression(Accuracy):
    """Calculates the accuracy for regression data.

    Used for regression tasks where predictions are continuous values.  Compares predicted values to true values within a certain precision.
    """

    def __init__(self):
        """Initializes the AccuracyRegression object with precision set to None.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.precision = None

    def init(self, y, reinit=False):
        """Initializes the precision value.

        Parameters
        ----------
        y : numpy.ndarray
            The true labels.
        reinit : bool, optional
            Whether to reinitialize the precision, by default False.

        Returns
        -------
        None
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """Compares predictions to true labels based on precision.

        Parameters
        ----------
        predictions : numpy.ndarray
            The predicted labels.
        y : numpy.ndarray
            The true labels.

        Returns
        -------
        numpy.ndarray
            A boolean array indicating whether each prediction is within the precision range.
        """
        return np.absolute(predictions - y) < self.precision

    def __str__(self):
        """
        Returns a string representation of the AccuracyRegression object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The string representation of the object, including the precision value.
        """
        return f"Accuracy_Regression: Precision(): {self.precision}"


AccuracyTypes = Union[AccuracyCategorical, AccuracyRegression]
