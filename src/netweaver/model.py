import copy
import pickle

import numpy as np
from typing import Tuple, Optional

from numpy.typing import ArrayLike

from netweaver.activation_layers import ActivationSoftmax, ActivationTypes
from netweaver.layers import LayerInput, LayerTypes, TrainableLayerTypes
from netweaver.lossfunctions import LossCategoricalCrossentropy, LossTypes
from netweaver.softmaxCCEloss import ActivationSoftmaxLossCategoricalCrossentropy
from netweaver.accuracy import AccuracyTypes
from netweaver.optimizers import OptimizerTypes

from tqdm.auto import tqdm

Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class Model:
    """
    A class representing a neural network model. It supports adding layers,
    setting loss, optimizer, and accuracy objects, training the model, and
    saving/loading parameters and the model itself.
    """

    def __init__(self) -> None:
        """
        Initialize the model with an empty list of layers and no softmax classifier output.
        """
        self.layers: list[LayerTypes | ActivationTypes] = []
        self.softmax_classifier_output = None  # 99

    def add(
        self,
        layer: LayerTypes | ActivationTypes,
    ) -> None:
        """Adds a layer to the model.

        This method appends the given layer to the model's list of layers.
        The layer can be either a Layer object or an Activation object.  
        **Available layers**: [LayerDense | LayerDropout | LayerInput]  
        **Available activations**: [ActivationReLU | ActivationSoftmax | ActivationSigmoid | ActivationLinear]

        Parameters
        ----------
        layer : LayerTypes | ActivationTypes
            The layer to be added to the model.
        """
        self.layers.append(layer)

    def set(
        self,
        *,
        loss: Optional[LossTypes] = None,
        optimizer: Optional[OptimizerTypes] = None,
        accuracy: Optional[AccuracyTypes] = None,
    ) -> None:
        """Sets the loss, optimizer, and accuracy objects for the model.

        Parameters are keyword-only arguments.

        Parameters
        ----------
        loss : LossTypes, optional
            The loss function to use, by default None.
        optimizer : OptimizerTypes, optional
            The optimizer to use, by default None.
        accuracy : AccuracyTypes, optional
            The accuracy metric to use, by default None.
        """
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self) -> None:
        """
        Finalizes the model architecture by chaining layers, identifying trainable layers,
        and setting up a softmax_classifier_output if applicable.

        1. Creates doubly linked list of layers
        2. Creates list of trainable layers
        3. Creates refrence for Output_Activation_Layer
        4. Passes the trainable layer to Loss Object, to facilate the compute of regularization loss
        5. Modifies object attribute self.softmax_classifier_output to refrence ActivationSoftmaxLossCategoricalCrossentropy() object
           if the model is multi-class classifier
        6. LayerInput x (LayerDense  x LayerActivations x LayerDropouts x LayerOuputActivation) x LayerLoss

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.input_layer = LayerInput()
        self.trainable_layers: list[TrainableLayerTypes] = []
        layer_count = len(self.layers)

        for i in range(layer_count):
            # doubly linked list to facilate both forward and backward pass
            self.layers[i].prev = self.input_layer if i == 0 else self.layers[i - 1]
            self.layers[i].next = (
                self.loss if i == layer_count - 1 else self.layers[i + 1]
            )
            # Identify trainable layers
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # Set the output layer activation
        self.output_layer_activation: ActivationTypes = self.layers[layer_count - 1]

        # facilate the compute of regularization loss by passing the self.trainable_layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # Set up softmax_classifier_output if applicable, for rapid backward progress
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(
            self.loss, LossCategoricalCrossentropy
        ):
            self.softmax_classifier_output = (
                ActivationSoftmaxLossCategoricalCrossentropy()
            )

    def train(
        self,
        X,
        y,
        *,
        epochs=1,
        batch_size=None,
        print_every=1,
        validation_data=None,
    ) -> None:
        """Trains the model on the provided data.

        Performs the training loop for a specified number of epochs, using mini-batches if specified.  Calculates loss and accuracy for each batch and epoch, and optionally performs validation.

        Parameters
        ----------
        X : numpy.ndarray
            Input training data.
        y : numpy.ndarray
            Target training labels.
        epochs : int, optional
            Number of training epochs, by default 1.
        batch_size : int, optional
            Size of training batches, by default None (full dataset).
        print_every : int, optional
            Frequency of printing training progress, by default 1.
        validation_data : tuple, optional
            Validation data as a tuple (X_val, y_val), by default None.

        Returns
        -------
        None

        1. Initialize AccuracyRegression's precision property. Using entire Y_data training data for calculation
        2. Modify train_steps value if batch_size value is not None
        3. Epoch: Resets accumulated loss and accuracy for every epoch run
           1. Batch: slice the batch
           2. Batch: Call to forward method with data and training=True (training argument mainly for dropout layers)
           3. Batch: Compute data_loss and R_loss
           4. Batch: Compute predictions using Output_Activation_Layer's prediction method
           5. Batch: Compute accuracy using predictions(from above step) and y_true
           6. Batch: Call to backward method with Output and y_true
           7. Batch: call to optimizer method with trainable layers, pre_update -> update_params -> post_update
           8. Batch: Loggin batch training progress
        5. Epoch: Compute Epoch's Loss and Accuracy using calculate_accumulated method
        6. Epoch: Logging Epoch summary
        7. Epoch: Validate the model using Validate data
        """
        self.accuracy.init(y)  # mainly for AccuracyRegression's precision. 

        train_steps = 1
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        for epoch in tqdm(range(1, epochs + 1), desc="Traning..."):
            self.loss.new_pass()  # for accumulated sum and count to compute loss over an epoch
            self.accuracy.new_pass()  # for accumulated sum and count for accuracies

            for step in tqdm(range(train_steps), desc=f"Epoch {epoch}", leave=False):
                # Get the current batch
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size : (step + 1) * batch_size]
                    batch_y = y[step * batch_size : (step + 1) * batch_size]

                # Perform forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(
                    output, batch_y, include_regularization=True
                )
                loss = data_loss + regularization_loss

                # Calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Update weights and biases
                self.optimizer_action()

                # Print Batch training progress
                if not step % print_every or step == train_steps - 1:
                    print(
                        f"step: {step:<10}"
                        + f"acc: {accuracy:<10.3f}"
                        + f"loss: {loss:<10.3f}"
                        + f"data_loss: {data_loss:<10.3f}"
                        + f"reg_loss: {regularization_loss:<10.3f}"
                        + f"lr: {self.optimizer.current_learning_rate:<10.7f}"
                    )

            # Print epoch summary
            epoch_data_loss, epoch_regularization_loss = (
                self.loss.calculate_accumulated(include_regularization=True)
            )
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(
                f"\nEpoch {epoch:<10f} summary:\n"
                + f"acc: {epoch_accuracy:<10.3f}"
                + f"loss: {epoch_loss:<10.3f}"
                + f"data_loss: {epoch_data_loss:<10.3f}"
                + f"reg_loss: {epoch_regularization_loss:<10.3f}"
                + f"lr: {self.optimizer.current_learning_rate:<10.7f}\n"
            )

            # Perform validation if validation data is provided
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    def forward(
        self,
        X: ArrayLike,
        training: bool,
    ) -> Float64Array2D:
        """
        Perform a forward pass through the model.

        Args:
            X: Input data.
            training: Whether the model is in training mode.

        Returns:
            Output of the last layer.
        """
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(
                layer.prev.output, training
            )  # the training argument is only consumed by Dropout layer

        return layer.output

    def backward(self, output, y):
        """
        Perform a backward pass through the model.

        Args:
            output: Output of the last layer.
            y: True labels.
        """
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def optimizer_action(self):
        self.optimizer.pre_update_params()
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

    def evaluate(self, X_val, y_val, *, batch_size=None):
        """
        Evaluate the model on validation data.

        Args:
            X_val: Validation input data.
            y_val: Validation true labels.
            batch_size: Batch size for evaluation.
        """
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        self.loss.new_pass()
        self.accuracy.new_pass()
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size : (step + 1) * batch_size]
                batch_y = y_val[step * batch_size : (step + 1) * batch_size]
            output = self.forward(
                batch_X, training=False
            )  # making dropout layers out of loop
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(
            "validation",
            +f"acc: {validation_accuracy:.3f}, " + f"loss: {validation_loss:.3f}",
        )

    def predict(self, X, *, batch_size=None):
        """
        Generate predictions for the given input data.

        Args:
            X: Input data.
            batch_size: Batch size for prediction.

        Returns:
            Predicted outputs.
        """
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size : (step + 1) * batch_size]
            batch_output = self.forward(
                batch_X, training=False
            )  # making dropout layer out of loop
            output.append(batch_output)
        return np.vstack(
            output
        )  # Each row is a batch's prediction. Each value in the row is the prediction of a sample in a batch

    def get_parameters(self):
        """
        Get the parameters (weights and biases) of all trainable layers.

        Returns:
            List of parameters for each trainable layer.
        """
        return [layer.get_parameters() for layer in self.trainable_layers]

    def save_parameters(self, path):
        """
        Save the parameters (weights and biases) of the model to a file.

        Args:
            path: Path to the file where parameters will be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(), f)

    def set_parameters(self, parameters):
        """
        Set the parameters (weights and biases) for all trainable layers.

        Args:
            parameters: List of parameters for each trainable layer.
        """
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def load_parameters(self, path):
        """
        Load the parameters (weights and biases) of the model from a file.

        Args:
            path: Path to the file from which parameters will be loaded.
        """
        with open(path, "rb") as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        """
        Save the entire model to a file.

        Args:
            path: Path to the file where the model will be saved.
        """
        model: Model = copy.deepcopy(self)
        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)
        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        """
        Load a model from a file.

        Args:
            path: Path to the file from which the model will be loaded.

        Returns:
            The loaded model.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
