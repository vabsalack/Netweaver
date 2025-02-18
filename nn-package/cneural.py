import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, TypeVar, Union
from numpy.typing import NDArray, DTypeLike, ArrayLike

"""
what it is?
where we can improvize?
why it is used?
how it works?
"""

class Layer_Input:
    """
    Layer_Input class represents the input layer of the neural network.
    """

    def forward(self, inputs: NDArray) -> None:
        """
        Performs a forward pass of the input layer.

        Parameters:
        inputs (NDArray): Input data.
        """
        self.output = inputs


class Layer_Dense:
    """
    what it is?
        * Dense of neurons in a layer. 
    where we can improvize?
        * weights initialization is one of crucial part in model convergence.
        * experiment with other initialization method such as he, xavier, etc.
    why it is used?
        * For creating a layer of neurons in a neural network.
    how it works?
        * init
        * forward method
        * backward method
    """

    def __init__(self, n_inputs: int, 
                 n_neurons: int,
                 weight_regularizer_L1: Union[float, int] = 0,
                 weight_regularizer_L2: Union[float, int] = 0, 
                 bias_regularizer_L1: Union[float, int] = 0,
                 bias_regularizer_L2: Union[float, int] = 0) -> None:
        """
        what it does?
            * Initialize weights and biases
            * sets regularization strength of L1 and L2
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))
        # L1 strength
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.bias_regularizer_L1 = bias_regularizer_L1
        # L2 strength
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L2 = bias_regularizer_L2

    def forward(self, inputs: NDArray) -> None:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues: NDArray) -> None:
        """
        what it does?
            * computes gradient of weights, biases and inputs
            * applies regularization on weights and biases gradients
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        # apply regularization
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

        
class Layer_Dropout:
    """
    what it is?
        * Dropout is a regularization technique where randomly selected neurons are ignored during training.
    where we can improvize?
        * dropout rate is one of the hyperparameter, experiment with different values.
    why it is used?
        * To prevent overfitting.
        * To improve generalization.
        * To make model robust.
        * To make model less sensitive to the specific weights of neurons.
    how it works?
        * init
        * forward method
        * backward method
        * Randomly sets some of the activations to zero during training by sampling from a binomial distribution.
        * This creates a dropout mask that deactivates certain neurons to prevent overfitting.
    """

    def __init__(self, rate: Union[float, int]) -> None:
        """
        what it does?
            * sets dropout rate
            * argument rate is the percentage of neurons to be deactivated.
            * self.rate = 1 - rate, is the percentage of neurons to be activated because numpy binomial method expects probability of success.
        """
        self.rate = 1 - rate

    def forward(self, 
                inputs: NDArray, 
                training: bool) -> None:
        """
        what it does?
            * creates binary mask only on training.
            * passes the input as it is during inference.
        """
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        self.output = self.inputs * self.binary_mask

    def backward(self, dvalues: NDArray) -> None:
        self.dinputs = dvalues * self.binary_mask


class Activation_ReLU:
    """
    what it is?
        * ReLu activation function is used to introduce non-linearity in the neural network.
    where we can improvize?
        * leaky ReLu, Parametric ReLu, Exponential ReLu, etc.
    why it is used?
        * To learn non-linear patterns in the data.
    how it works?
        * forward method
        * backward method
        * predictions method
        * Mathematically, ReLu(x) = max(0, x)
    """

    def forward(self, 
                inputs: NDArray, 
                training: bool) -> None:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: NDArray) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs: NDArray) -> NDArray:
        return outputs


class Activation_Softmax:
    """
    what it is?
        * Softmax activation function is used to convert raw scores into probabilities.
        * It is used in the output layer of a neural network for multi-class classification.
        * It squashes the raw scores into a range of [0, 1] and normalizes them.
    where we can improvize?
        * Softmax is the best activation function for multi-class classification.
    why it is used?
        * To convert raw scores into probabilities.
        * To make the model output interpretable.
    how it works?
        * forward method
        * backward method
        * predictions method
        * Mathematically, softmax(x) = exp(x) / sum(exp(x))
        * Exponentiation graph is rapid, avoid overflow by making maximum to zero. get the idea
    """

    def forward(self, 
                inputs: NDArray, 
                training: bool) -> None:
        """
        calculates softmax values of inputs and stores it in self.output
        """
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues: NDArray) -> None:
        """
        what it does?
            * creates self.dinputs beforehand since numpy array's aren't dynamically resizable.
            * calculates gradient of softmax values and stores it in self.dinputs
            * softmax derivative is calculated using jacobian matrix
            * jacobian matrix is a square matrix containing all first-order partial derivatives of a vector-valued function.
            * refer math behind back propogation for intuitive understanding
        """
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs: NDArray) -> NDArray:
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid:
    """
    what it is?
        * It is used in the output layer of a neural network for binary logistic regression
        * sigmod function only takes one input and returns one output.
    where we can improvize?
    why it is used?
        * To convert raw scores into probabilities.
        * To make the model output interpretable
    how it works?
        * forward method
        * backward method
        * predictions method
        * Mathematically, sigmoid(x) = 1 / (1 + exp(-x))
    """

    def forward(self, 
                inputs: NDArray, 
                training: NDArray) -> None:
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: NDArray) -> None:
        """
        what it does? 
            * The derivative of the sigmoid function is calculated as:
            * sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        """
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs: NDArray) -> NDArray:
        """
        what it does?
            * product of arr: NDArray[bool_] and x: int is y: NDArray[int64]
            * False values replaced by 0 and True values replaced by x
            * array([False, True]) * -2 = array([0, -2])
        """
        return (outputs > 0.5) * 1


class Activation_Linear:
    """
    what it is?
        * Linear activation function is used to pass the input directly to the output without any modification.
    where we can improvize?
        * This is a simple activation function, not much to improve.
    why it is used?
        * It is used in the output layer for regression tasks where we need to predict continuous values.
    how it works?
        * forward method
        * backward method
        * predictions method
        * It simply returns the input as the output.
    """

    def forward(self, 
                inputs: NDArray, 
                training: bool) -> None:
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues: NDArray) -> None:
        self.dinputs = dvalues.copy()

    def predictions(self, outputs: NDArray) -> NDArray:
        return outputs

class Loss:
    """
    what it is?
        * Base class for loss functions.
    where we can improvize?
        * Implement additional loss functions as needed.
    why it is used?
        * To calculate the difference between the predicted output and the actual output.
    how it works?
        * remember_trainable_layers method sets self.trainable_layers property.
        * regularization_loss method calculates the regularization loss using layers in self.trainable_layers.
        * calculate method calculates the data loss using child class's forward method and regularization loss using regularization_loss method.
        * It provides methods to calculate the loss and its gradient.
    """

    def remember_trainable_layers(self, trainable_layers: List[Union[Layer_Dense]]) -> None:
        self.trainable_layers = trainable_layers

    def regularization_loss(self) -> float:
        regularization_loss: float = .0
        for layer in self.trainable_layers:
            # L1 penalty
            if layer.weight_regularizer_L1 > 0:
                regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))
            if layer.bias_regularizer_L1 > 0:
                regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))
            # L2 penalty
            if layer.weight_regularizer_L2 > 0:
                regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_L2 > 0:
                regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

    def calculate(self, 
                  output: NDArray, 
                  y: NDArray, 
                  *, 
                  include_regularization: bool=False) -> Union[float, Tuple[float, float]]:
        """
        what it does?
            * calculates data and regularization loss
            * data loss is the mean of sample losses
            * regularization loss is the sum of L1 and L2 penalties
            * returns only data loss by default
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    
# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
    # Backward pass
    def backward(self, dvalues, y_true):
        # davalues are softmax outputs (predicted outputs)
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient, true lables are 1
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        # np.log(1e-323) = -inf
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 -y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    # backward pass
    def backward(self, dvalues, y_true):
        # here dvalues are outputs from sigmoid
        samples = len(dvalues)
        # number of output in every sample
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true)/(1 - clipped_dvalues)) / outputs
        # normalize gradient
        self.dinputs /= samples

# mean squared error loss
class Loss_MeanSquaredError(Loss):
    # forward pass
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        # number of outputs in every sample
        outputs = len(dvalues[0])

        # gradients on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # normalize gradient
        self.dinputs = self.dinputs / samples

# mean absolute error loss
class Loss_MeanAbsoluteError(Loss):
    #forward pass
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    # backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # nubmer of outputs in every sample
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        # normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            # adding 1 ensure value decreases
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, Layer):

        # If we use momentum
        if self.momentum:
            if not hasattr(Layer, "weight_momentums"):
                Layer.weight_momentums = np.zeros_like(Layer.weights)
                Layer.bias_momentums = np.zeros_like(Layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * Layer.weight_momentums - self.current_learning_rate * Layer.dweights
            Layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = self.momentum * Layer.bias_momentums - self.current_learning_rate * Layer.dbiases
            Layer.bias_momentums = bias_updates
            # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * Layer.dweights
            bias_updates = -self.current_learning_rate * Layer.dbiases

                    
        Layer.weights += weight_updates
        Layer.biases += bias_updates

    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    # learning rate of 1. is default for this optimizer
    # epsilon for numerical stability
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            # adding 1 ensure value < 0
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, Layer):

        if not hasattr(Layer, "weight_cache"):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)
        
        # update cache with squared current gradients
        Layer.weight_cache += Layer.dweights**2
        Layer.bias_cache += Layer.dbiases**2

        # vanilla + normalization with square rooted cache
        Layer.weights += -self.current_learning_rate * Layer.dweights / (np.sqrt(Layer.weight_cache) + self.epsilon)        
        Layer.biases += -self.current_learning_rate * Layer.dbiases / (np.sqrt(Layer.bias_cache) + self.epsilon)        


    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    # learning rate of 0.001 is default for this optimizer
    # epsilon for numerical stability
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta = beta

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            # adding 1 ensure value < 0
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, Layer):

        if not hasattr(Layer, "weight_cache"):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)
        
        # update cache with squared current gradients
        # exponentially discounts past gradients
        # EWMA (Exponentially Weighted Moving Average)
        Layer.weight_cache = self.beta * Layer.weight_cache + (1 - self.beta) * Layer.dweights**2
        Layer.bias_cache += self.beta * Layer.bias_cache + (1- self.beta) * Layer.dbiases**2

        # vanilla + normalization with square rooted cache
        Layer.weights += -self.current_learning_rate * Layer.dweights / (np.sqrt(Layer.weight_cache) + self.epsilon)        
        Layer.biases += -self.current_learning_rate * Layer.dbiases / (np.sqrt(Layer.bias_cache) + self.epsilon)        


    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    # learning rate, beta_1 and beta_2 are in default values
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1        
        self.beta_2 = beta_2     

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, Layer):
        
        if not hasattr(Layer, "weight_cache"):
            Layer.weight_momentums = np.zeros_like(Layer.weights)
            Layer.bias_momentums = np.zeros_like(Layer.biases)
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)

        # scaled momentums (EWMA)
        Layer.weight_momentums = self.beta_1 * Layer.weight_momentums + (1 - self.beta_1) * Layer.dweights
        Layer.bias_momentums = self.beta_1 * Layer.bias_momentums + (1 - self.beta_1) * Layer.dbiases
        # bias correction
        weight_momentums_corrected = Layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = Layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # scaled gradients (EWMA)
        Layer.weight_cache = self.beta_2 * Layer.weight_cache + (1 - self.beta_2) * Layer.dweights**2
        Layer.bias_cache = self.beta_2 * Layer.bias_cache + (1 - self.beta_2) * Layer.dbiases**2
        # bias correction
        weight_cache_corrected = Layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = Layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        # params updates
        Layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        Layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)


    def post_update_params(self):
        self.iterations += 1








           


    

