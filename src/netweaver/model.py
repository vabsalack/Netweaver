import copy
import pickle
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from netweaver.accuracy import AccuracyTypes
from netweaver.activation_layers import ActivationSoftmax, ActivationTypes
from netweaver.layers import LayerInput, LayerTypes, TrainableLayerTypes
from netweaver.lossfunctions import LossCategoricalCrossentropy, LossTypes
from netweaver.optimizers import OptimizerTypes
from netweaver.softmax_cce_loss import ActivationSoftmaxLossCategoricalCrossentropy

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

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.layers: list[LayerTypes | ActivationTypes] = []
        self.softmax_classifier_output = None  # 99

    def add(
        self,
        layer: LayerTypes | ActivationTypes,
    ) -> None:
        """Adds a layer to the model.

        Parameters
        ----------
        layer : LayerTypes | ActivationTypes
            The layer to add. Can be an instance of LayerDense, LayerDropout, or any activation layer.

        Returns
        -------
        None
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

        Returns
        -------
        None
        """
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self) -> None:
        """Finalizes the model architecture.

        Prepares the model for training by performing the following steps:

        1. Creates a doubly linked list of layers to facilitate forward and backward passes.
        2. Identifies and stores trainable layers (those with weights).
        3. Sets a reference to the output layer's activation function.
        4. Provides the loss function with a reference to the trainable layers for regularization.
        5. Creates a combined softmax activation and categorical cross-entropy loss object if applicable for optimization.

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
            self.layers[i].next = self.loss if i == layer_count - 1 else self.layers[i + 1]
            # Identify trainable layers
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # Set the output layer activation
        self.output_layer_activation: ActivationTypes = self.layers[layer_count - 1]

        # facilate the compute of regularization loss by passing the self.trainable_layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # Set up softmax_classifier_output if applicable, for rapid backward progress
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, LossCategoricalCrossentropy):
            self.softmax_classifier_output = ActivationSoftmaxLossCategoricalCrossentropy()

    def train(
        self,
        instances_train,
        gtruth_train,
        *,
        epochs=1,
        batch_size=None,
        print_every=1,
        validation_data=None,
    ) -> None:
        """Trains the model using the provided data and parameters.

        This method iterates through the training data in batches (or uses the entire dataset if batch_size is None),
        performs forward and backward passes, updates model parameters using the optimizer, and prints training progress.
        It also performs validation at the end of each epoch if validation data is provided.

        Parameters
        ----------
        instances_train : ArrayLike
            Training data.
        gtruth_train : ArrayLike
            Training labels.
        epochs : int, optional (keyword-only argument)
            Number of training epochs, by default 1.
        batch_size : int, optional (keyword-only argument)
            Size of each mini-batch, by default None (uses the entire dataset).
        print_every : int, optional (keyword-only argument)
            Frequency of printing training progress (in steps), by default 1.
        validation_data : tuple, optional (keyword-only argument)
            Validation data as a tuple (instances_validation, gtruth_validation), by default None.

        Returns
        -------
        None
        """
        self.accuracy.init(gtruth_train)
        train_steps = 1
        if batch_size is not None:
            train_steps = len(instances_train) // batch_size
            if train_steps * batch_size < len(instances_train):
                train_steps += 1

        for epoch in tqdm(range(1, epochs + 1), desc="Traning..."):
            self.loss.new_pass()  # for accumulated sum and count to compute loss over an epoch
            self.accuracy.new_pass()  # for accumulated sum and count for accuracies

            for step in tqdm(range(train_steps), desc=f"Epoch {epoch}", leave=False):
                # Get the current batch
                if batch_size is None:
                    batch_instances_train = instances_train
                    batch_gtruth_train = gtruth_train
                else:
                    batch_instances_train = instances_train[step * batch_size : (step + 1) * batch_size]
                    batch_gtruth_train = gtruth_train[step * batch_size : (step + 1) * batch_size]

                # Perform forward pass
                output = self.forward(batch_instances_train, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_gtruth_train, include_regularization=True)
                loss = data_loss + regularization_loss

                # Calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_gtruth_train)

                # Perform backward pass
                self.backward(output, batch_gtruth_train)

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
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
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
        instances: ArrayLike,
        training: bool,
    ) -> Float64Array2D:
        """
        Perform a forward pass through the model.

        Args:
            instances: Input data.
            training: Whether the model is in training mode.

        Returns:
            Output of the last layer.
        """
        self.input_layer.forward(instances, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)  # the training argument is only consumed by Dropout layer

        return layer.output

    def backward(self, output, gtruth):
        """
        Perform a backward pass through the model.

        Args:
            output: Output of the last layer.
            gtruth: True labels.
        """
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, gtruth)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, gtruth)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def optimizer_action(self):
        self.optimizer.pre_update_params()
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

    def evaluate(self, instances_validation, gtruth_validation, *, batch_size=None):
        """
        Evaluate the model on validation data.

        Args:
            instances_validation: Validation input data.
            gtruth_validation: Validation true labels.
            batch_size: Batch size for evaluation.
        """
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(instances_validation) // batch_size
            if validation_steps * batch_size < len(instances_validation):
                validation_steps += 1
        self.loss.new_pass()
        self.accuracy.new_pass()
        for step in range(validation_steps):
            if batch_size is None:
                batch_instances_validation = instances_validation
                batch_gtruth_validation = gtruth_validation
            else:
                batch_instances_validation = instances_validation[step * batch_size : (step + 1) * batch_size]
                batch_gtruth_validation = gtruth_validation[step * batch_size : (step + 1) * batch_size]
            output = self.forward(batch_instances_validation, training=False)  # making dropout layers out of the loop
            self.loss.calculate(output, batch_gtruth_validation)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_gtruth_validation)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(
            "validation",
            +f"acc: {validation_accuracy:.3f}, " + f"loss: {validation_loss:.3f}",
        )

    def predict(self, instances_prediction, *, batch_size=None):
        """
        Generate predictions for the given input data.

        Args:
            instances_prediction: Input data.
            batch_size: Batch size for prediction.

        Returns:
            Predicted outputs.
        """
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(instances_prediction) // batch_size
            if prediction_steps * batch_size < len(instances_prediction):
                prediction_steps += 1
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_instances_prediction = instances_prediction
            else:
                batch_instances_prediction = instances_prediction[step * batch_size : (step + 1) * batch_size]
            batch_output = self.forward(batch_instances_prediction, training=False)  # making dropout layer out of loop
            output.append(batch_output)
        return np.vstack(output)  # Each row is a batch's prediction. Each value in the row is the prediction of a sample in a batch

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
            return pickle.load(f)
